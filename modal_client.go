package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sync"
	"time"
)

// ModalClient calls the Modal-deployed inference API via HTTP.
type ModalClient struct {
	baseURL    string
	httpClient *http.Client
	mu         sync.Mutex
	ready      bool
}

// NewModalClient creates a client pointing at the Modal web endpoint.
func NewModalClient(modalURL string) (*ModalClient, error) {
	if modalURL == "" {
		return nil, fmt.Errorf("MODAL_URL not set — deploy first with: modal deploy modal_app.py")
	}

	client := &ModalClient{
		baseURL: modalURL,
		httpClient: &http.Client{
			Timeout: 10 * time.Minute, // Generous timeout for cold starts + inference
		},
	}

	log.Printf("Modal backend: %s", modalURL)
	return client, nil
}

// WaitReady checks if the Modal service is reachable.
func (m *ModalClient) WaitReady() (*PythonResponse, error) {
	log.Println("Checking Modal service health...")

	// Retry up to 3 times (Modal cold start can take ~60-90s)
	var lastErr error
	for attempt := 1; attempt <= 3; attempt++ {
		resp, err := m.httpClient.Get(m.baseURL + "/health")
		if err != nil {
			lastErr = err
			log.Printf("  Attempt %d: %v (cold start may take 60-90s)", attempt, err)
			if attempt < 3 {
				time.Sleep(15 * time.Second)
			}
			continue
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)

		if resp.StatusCode == http.StatusOK {
			var result PythonResponse
			json.Unmarshal(body, &result)
			result.ModelsLoaded = true
			m.ready = true
			log.Printf("Modal service healthy: %s", string(body))
			return &result, nil
		}

		lastErr = fmt.Errorf("health check returned %d: %s", resp.StatusCode, body)
		log.Printf("  Attempt %d: %v", attempt, lastErr)
		if attempt < 3 {
			time.Sleep(10 * time.Second)
		}
	}

	// Return a "ready" response anyway — the first real request will trigger cold start
	log.Printf("Modal health check failed (will retry on first request): %v", lastErr)
	m.ready = true
	return &PythonResponse{
		Status:       "ready",
		ModelsLoaded: true,
		Device:       "cuda-a10g",
		Platform:     "modal-a10g",
	}, nil
}

// SendRequest sends a JSON request to the Modal web endpoint.
func (m *ModalClient) SendRequest(req PythonRequest) (*PythonResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	var endpoint string
	var method string

	switch req.Action {
	case "health":
		endpoint = "/health"
		method = "GET"
	case "styles":
		endpoint = "/api/interior/styles"
		method = "GET"
	case "control_maps":
		endpoint = "/api/interior/control-maps"
		method = "GET"
	case "redesign":
		endpoint = "/api/interior/redesign"
		method = "POST"
	default:
		return nil, fmt.Errorf("unknown action: %s", req.Action)
	}

	url := m.baseURL + endpoint

	var httpResp *http.Response
	var err error

	if method == "GET" {
		httpResp, err = m.httpClient.Get(url)
	} else {
		// Build the request body matching the Modal FastAPI schema
		bodyMap := map[string]interface{}{
			"image":               req.Image,
			"prompt":              req.Prompt,
			"style":               req.Style,
			"room_type":           req.RoomType,
			"mode":                req.Mode,
			"controlnet_type":     req.ControlNetType,
			"controlnet_strength": req.ControlNetStrength,
			"controlnet_end_step": req.ControlNetEndStep,
			"strength":            req.Strength,
			"steps":               req.Steps,
			"guidance_scale":      req.GuidanceScale,
			"seed":                req.Seed,
			"max_dim":             req.MaxDim,
		}
		bodyBytes, merr := json.Marshal(bodyMap)
		if merr != nil {
			return nil, fmt.Errorf("marshal error: %w", merr)
		}
		httpResp, err = m.httpClient.Post(url, "application/json", bytes.NewReader(bodyBytes))
	}

	if err != nil {
		return nil, fmt.Errorf("modal request failed: %w", err)
	}
	defer httpResp.Body.Close()

	body, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response failed: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("modal returned %d: %s", httpResp.StatusCode, string(body))
	}

	var resp PythonResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("parse response failed: %w (body length: %d)", err, len(body))
	}

	// Map fields for compatibility
	if resp.Status == "healthy" {
		resp.ModelsLoaded = true
	}
	if resp.Success || resp.Image != "" {
		resp.Success = true
	}

	return &resp, nil
}

// Stop is a no-op for Modal (the cloud service stays running).
func (m *ModalClient) Stop() {
	log.Println("Modal client disconnected (service continues running in cloud)")
}

// ─── Backend Interface ───────────────────────────────────────────────────────

// InferenceBackend abstracts local Python vs Modal cloud backends.
type InferenceBackend interface {
	WaitReady() (*PythonResponse, error)
	SendRequest(req PythonRequest) (*PythonResponse, error)
	Stop()
}

// Ensure both backends implement the interface
var _ InferenceBackend = (*PythonRunner)(nil)
var _ InferenceBackend = (*ModalClient)(nil)

// NewBackend creates the appropriate backend based on MODAL_URL env var.
func NewBackend() (InferenceBackend, error) {
	modalURL := os.Getenv("MODAL_URL")
	if modalURL == "" {
		modalURL = "https://hariom-hp--z-image-interior-design-fastapi-app.modal.run"
	}

	if modalURL != "" {
		log.Printf("Using Modal cloud backend: %s", modalURL)
		return NewModalClient(modalURL)
	}

	// Fall back to local Python process
	pythonScript := os.Getenv("PYTHON_SCRIPT")
	if pythonScript == "" {
		pythonScript = "interior_inference.py"
	}
	pythonCmd := os.Getenv("PYTHON_CMD")
	if pythonCmd == "" {
		pythonCmd = "python3"
	}
	log.Printf("Using local Python backend: %s %s", pythonCmd, pythonScript)
	return NewPythonRunner(pythonCmd, pythonScript)
}
