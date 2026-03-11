package main

import (
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// Create backend (Modal cloud or local Python — auto-detected via MODAL_URL)
	backend, err := NewBackend()
	if err != nil {
		log.Fatalf("Failed to start inference backend: %v", err)
	}
	defer backend.Stop()

	// Wait for backend to become ready
	log.Println("Waiting for inference service to be ready...")
	readyResp, err := backend.WaitReady()
	if err != nil {
		log.Fatalf("Inference service failed to start: %v", err)
	}
	log.Printf("Service ready: %v", readyResp)

	// Set up Gin router
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()

	// CORS for Flutter app
	r.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"*"},
		AllowMethods:     []string{"GET", "POST", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		AllowCredentials: true,
	}))

	// Routes — handlers use the InferenceBackend interface
	r.GET("/health", HealthHandlerV2(backend))
	r.POST("/api/interior/redesign", RedesignHandlerV2(backend))
	r.GET("/api/interior/styles", StylesHandlerV2(backend))
	r.GET("/api/interior/control-maps", ControlMapsHandlerV2(backend))

	// Graceful shutdown
	go func() {
		quit := make(chan os.Signal, 1)
		signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
		<-quit
		log.Println("Shutting down...")
		backend.Stop()
		os.Exit(0)
	}()

	backendType := "local-python"
	if os.Getenv("MODAL_URL") != "" {
		backendType = "modal-l4-gpu"
	}

	log.Printf("Z-Image Interior Design API (ControlNet Edition) on port %s", port)
	log.Printf("Backend: %s", backendType)
	log.Printf("Endpoints:")
	log.Printf("  GET  /health")
	log.Printf("  POST /api/interior/redesign")
	log.Printf("  GET  /api/interior/styles")
	log.Printf("  GET  /api/interior/control-maps")

	if err := http.ListenAndServe(":"+port, r); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
