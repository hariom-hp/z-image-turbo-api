package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os/exec"
	"sync"
	"time"
)

// PythonRunner manages the long-running Python inference process.
type PythonRunner struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Scanner
	stderr io.ReadCloser
	mu     sync.Mutex
	ready  bool
}

// NewPythonRunner starts the Python inference process.
func NewPythonRunner(pythonCmd, scriptPath string) (*PythonRunner, error) {
	cmd := exec.Command(pythonCmd, "-u", scriptPath)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stdin pipe: %w", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stdout pipe: %w", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stderr pipe: %w", err)
	}

	runner := &PythonRunner{
		cmd:    cmd,
		stdin:  stdin,
		stdout: bufio.NewScanner(stdout),
		stderr: stderr,
	}

	// Increase scanner buffer for large base64 images (100MB)
	runner.stdout.Buffer(make([]byte, 0, 64*1024), 100*1024*1024)

	// Forward Python stderr to Go's log
	go runner.forwardStderr()

	log.Printf("Starting Python: %s -u %s", pythonCmd, scriptPath)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start Python process: %w", err)
	}

	return runner, nil
}

// WaitReady waits for the Python process to signal it's ready.
func (r *PythonRunner) WaitReady() (*PythonResponse, error) {
	// Wait for the initial "ready" message with a timeout
	done := make(chan *PythonResponse, 1)
	errChan := make(chan error, 1)

	go func() {
		if r.stdout.Scan() {
			line := r.stdout.Text()
			var resp PythonResponse
			if err := json.Unmarshal([]byte(line), &resp); err != nil {
				errChan <- fmt.Errorf("failed to parse ready response: %w (line: %s)", err, line)
				return
			}
			done <- &resp
		} else {
			if err := r.stdout.Err(); err != nil {
				errChan <- fmt.Errorf("stdout scanner error: %w", err)
			} else {
				errChan <- fmt.Errorf("Python process closed stdout before sending ready signal")
			}
		}
	}()

	// 120 minute timeout for model loading (Z-Image-Turbo is ~12GB, first download can be slow)
	select {
	case resp := <-done:
		r.ready = resp.ModelsLoaded
		return resp, nil
	case err := <-errChan:
		return nil, err
	case <-time.After(120 * time.Minute):
		return nil, fmt.Errorf("timeout waiting for Python service (120 min)")
	}
}

// SendRequest sends a JSON request to Python and waits for the response.
func (r *PythonRunner) SendRequest(req PythonRequest) (*PythonResponse, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Marshal request to JSON
	reqBytes, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Send to stdin
	_, err = fmt.Fprintf(r.stdin, "%s\n", reqBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to write to Python stdin: %w", err)
	}

	// Read response from stdout (with timeout)
	respChan := make(chan string, 1)
	errChan := make(chan error, 1)

	go func() {
		if r.stdout.Scan() {
			respChan <- r.stdout.Text()
		} else {
			if err := r.stdout.Err(); err != nil {
				errChan <- fmt.Errorf("stdout scanner error: %w", err)
			} else {
				errChan <- fmt.Errorf("Python process closed stdout")
			}
		}
	}()

	// 5 minute timeout for inference
	timeout := 5 * time.Minute
	if req.Action == "health" || req.Action == "styles" {
		timeout = 10 * time.Second
	}

	select {
	case line := <-respChan:
		var resp PythonResponse
		if err := json.Unmarshal([]byte(line), &resp); err != nil {
			return nil, fmt.Errorf("failed to parse python response: %w (line length: %d)", err, len(line))
		}
		return &resp, nil
	case err := <-errChan:
		return nil, err
	case <-time.After(timeout):
		return nil, fmt.Errorf("timeout waiting for Python response (%v)", timeout)
	}
}

// forwardStderr reads Python's stderr and logs it.
func (r *PythonRunner) forwardStderr() {
	scanner := bufio.NewScanner(r.stderr)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		log.Printf("[Python] %s", scanner.Text())
	}
}

// Stop gracefully stops the Python process.
func (r *PythonRunner) Stop() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.cmd.Process != nil {
		// Try graceful shutdown first
		quitReq, _ := json.Marshal(PythonRequest{Action: "quit"})
		fmt.Fprintf(r.stdin, "%s\n", quitReq)

		// Wait a bit for graceful exit
		done := make(chan error, 1)
		go func() {
			done <- r.cmd.Wait()
		}()

		select {
		case <-done:
			log.Println("Python process stopped gracefully")
		case <-time.After(5 * time.Second):
			log.Println("Python process didn't stop gracefully, killing...")
			r.cmd.Process.Kill()
		}
	}
}
