package main

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

// ─── Request/Response Types ──────────────────────────────────────────────────

// RedesignRequest represents the incoming room redesign request.
type RedesignRequest struct {
	Image              string  `json:"image" binding:"required"` // Base64 encoded room photo
	Prompt             string  `json:"prompt"`                   // User's design prompt
	Style              string  `json:"style"`                    // modern, minimalist, luxury, etc.
	RoomType           string  `json:"room_type"`                // living_room, bedroom, kitchen, etc.
	Mode               string  `json:"mode"`                     // "auto", "controlnet", or "img2img"
	ControlNetType     string  `json:"controlnet_type"`          // depth_v3, canny, hed, or auto
	ControlNetStrength float64 `json:"controlnet_strength"`      // 0.0-1.0, how strongly control map guides (default 1.0)
	ControlNetEndStep  int     `json:"controlnet_end_step"`      // When to stop ControlNet (default 5 of 9)
	Strength           float64 `json:"strength"`                 // 0.0-1.0, img2img strength (default 0.75)
	Steps              int     `json:"steps"`                    // Inference steps (default 9 for Turbo)
	GuidanceScale      float64 `json:"guidance_scale"`           // CFG scale (default 0.0 for Turbo)
	Seed               int     `json:"seed"`                     // -1 for random
	MaxDim             int     `json:"max_dim"`                  // Max image dimension (default 1024)
}

// RedesignResponse is the API response for room redesign.
type RedesignResponse struct {
	Success            bool    `json:"success"`
	Image              string  `json:"image,omitempty"` // Base64 encoded result
	Width              int     `json:"width,omitempty"`
	Height             int     `json:"height,omitempty"`
	GenerationTimeMs   int64   `json:"generation_time_ms,omitempty"`
	Seed               int     `json:"seed,omitempty"`
	PromptUsed         string  `json:"prompt_used,omitempty"`
	Strength           float64 `json:"strength,omitempty"`
	Device             string  `json:"device,omitempty"`
	ControlMapUsed     string  `json:"control_map_used,omitempty"` // depth_v3, canny, hed, none
	Mode               string  `json:"mode,omitempty"`             // controlnet or img2img
	ControlNetEndStep  int     `json:"controlnet_end_step,omitempty"`
	ControlNetStrength float64 `json:"controlnet_strength,omitempty"`
	Error              string  `json:"error,omitempty"`
}

// PythonRequest is the JSON sent to the Python process via stdin.
type PythonRequest struct {
	Action             string  `json:"action"`
	Image              string  `json:"image,omitempty"`
	Prompt             string  `json:"prompt,omitempty"`
	Style              string  `json:"style,omitempty"`
	RoomType           string  `json:"room_type,omitempty"`
	Mode               string  `json:"mode,omitempty"`                // auto, controlnet, img2img
	ControlNetType     string  `json:"controlnet_type,omitempty"`     // depth_v3, canny, hed, auto
	ControlNetStrength float64 `json:"controlnet_strength,omitempty"` // 0.0-1.0
	ControlNetEndStep  int     `json:"controlnet_end_step,omitempty"` // 0-9
	Strength           float64 `json:"strength,omitempty"`
	Steps              int     `json:"steps,omitempty"`
	GuidanceScale      float64 `json:"guidance_scale,omitempty"`
	Seed               int     `json:"seed,omitempty"`
	MaxDim             int     `json:"max_dim,omitempty"`
}

// PythonResponse is the JSON received from the Python process via stdout.
type PythonResponse struct {
	Status             string   `json:"status,omitempty"`
	Success            bool     `json:"success,omitempty"`
	ModelsLoaded       bool     `json:"models_loaded,omitempty"`
	Device             string   `json:"device,omitempty"`
	Platform           string   `json:"platform,omitempty"`
	Image              string   `json:"image,omitempty"`
	Width              int      `json:"width,omitempty"`
	Height             int      `json:"height,omitempty"`
	GenerationTimeMs   int64    `json:"generation_time_ms,omitempty"`
	Seed               int      `json:"seed,omitempty"`
	PromptUsed         string   `json:"prompt_used,omitempty"`
	Strength           float64  `json:"strength,omitempty"`
	ControlMapUsed     string   `json:"control_map_used,omitempty"`
	Mode               string   `json:"mode,omitempty"`
	ControlNetEndStep  int      `json:"controlnet_end_step,omitempty"`
	ControlNetStrength float64  `json:"controlnet_strength,omitempty"`
	ControlNetAvail    bool     `json:"controlnet_available,omitempty"`
	DepthAvail         bool     `json:"depth_available,omitempty"`
	Error              string   `json:"error,omitempty"`
	Styles             []string `json:"styles,omitempty"`
	RoomTypes          []string `json:"room_types,omitempty"`
}

// ─── Handlers ────────────────────────────────────────────────────────────────

// HealthHandler returns the health status (legacy — uses PythonRunner directly).
func HealthHandler(runner *PythonRunner) gin.HandlerFunc {
	return func(c *gin.Context) {
		resp, err := runner.SendRequest(PythonRequest{Action: "health"})
		if err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"status":  "unhealthy",
				"error":   err.Error(),
				"service": "python_inference",
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"status":               "healthy",
			"models_loaded":        resp.ModelsLoaded,
			"device":               resp.Device,
			"service":              "z-image-turbo-controlnet",
			"controlnet_available": resp.ControlNetAvail,
			"depth_available":      resp.DepthAvail,
		})
	}
}

// RedesignHandler handles room redesign requests.
func RedesignHandler(runner *PythonRunner) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req RedesignRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, RedesignResponse{
				Success: false,
				Error:   "Invalid request: " + err.Error(),
			})
			return
		}

		// Validate: need at least image + (prompt or style or room_type)
		if req.Prompt == "" && req.Style == "" && req.RoomType == "" {
			c.JSON(http.StatusBadRequest, RedesignResponse{
				Success: false,
				Error:   "Provide at least one of: prompt, style, room_type",
			})
			return
		}

		// Apply defaults
		if req.Mode == "" {
			req.Mode = "auto"
		}
		if req.ControlNetType == "" {
			req.ControlNetType = "depth_v3"
		}
		if req.ControlNetStrength == 0 {
			req.ControlNetStrength = 1.0
		}
		if req.ControlNetEndStep == 0 {
			req.ControlNetEndStep = 5
		}
		if req.Strength == 0 {
			req.Strength = 0.75
		}
		if req.Steps == 0 {
			req.Steps = 9
		}
		// guidance_scale default is 0.0 for Turbo (no CFG)
		if req.Seed == 0 {
			req.Seed = -1
		}
		if req.MaxDim == 0 {
			req.MaxDim = 1024
		}

		// Send to Python
		startTime := time.Now()
		pyReq := PythonRequest{
			Action:             "redesign",
			Image:              req.Image,
			Prompt:             req.Prompt,
			Style:              req.Style,
			RoomType:           req.RoomType,
			Mode:               req.Mode,
			ControlNetType:     req.ControlNetType,
			ControlNetStrength: req.ControlNetStrength,
			ControlNetEndStep:  req.ControlNetEndStep,
			Strength:           req.Strength,
			Steps:              req.Steps,
			GuidanceScale:      req.GuidanceScale,
			Seed:               req.Seed,
			MaxDim:             req.MaxDim,
		}

		resp, err := runner.SendRequest(pyReq)
		if err != nil {
			c.JSON(http.StatusInternalServerError, RedesignResponse{
				Success: false,
				Error:   "Inference failed: " + err.Error(),
			})
			return
		}

		if !resp.Success {
			c.JSON(http.StatusInternalServerError, RedesignResponse{
				Success: false,
				Error:   resp.Error,
			})
			return
		}

		totalTime := time.Since(startTime).Milliseconds()

		c.JSON(http.StatusOK, RedesignResponse{
			Success:            true,
			Image:              resp.Image,
			Width:              resp.Width,
			Height:             resp.Height,
			GenerationTimeMs:   totalTime,
			Seed:               resp.Seed,
			PromptUsed:         resp.PromptUsed,
			Strength:           resp.Strength,
			Device:             resp.Device,
			ControlMapUsed:     resp.ControlMapUsed,
			Mode:               resp.Mode,
			ControlNetEndStep:  resp.ControlNetEndStep,
			ControlNetStrength: resp.ControlNetStrength,
		})
	}
}

// StylesHandler returns available interior design styles and room types.
func StylesHandler(runner *PythonRunner) gin.HandlerFunc {
	return func(c *gin.Context) {
		resp, err := runner.SendRequest(PythonRequest{Action: "styles"})
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Failed to get styles: " + err.Error(),
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"styles":     resp.Styles,
			"room_types": resp.RoomTypes,
		})
	}
}

// ControlMapsHandler returns available control map types for ControlNet.
func ControlMapsHandler(runner *PythonRunner) gin.HandlerFunc {
	return func(c *gin.Context) {
		resp, err := runner.SendRequest(PythonRequest{Action: "control_maps"})
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Failed to get control maps: " + err.Error(),
			})
			return
		}

		c.JSON(http.StatusOK, resp)
	}
}

// ─── V2 Handlers (InferenceBackend interface — works with Modal & local) ────

// HealthHandlerV2 returns health status via any backend.
func HealthHandlerV2(backend InferenceBackend) gin.HandlerFunc {
	return func(c *gin.Context) {
		resp, err := backend.SendRequest(PythonRequest{Action: "health"})
		if err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"status": "unhealthy",
				"error":  err.Error(),
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"status":               "healthy",
			"models_loaded":        resp.ModelsLoaded,
			"device":               resp.Device,
			"service":              "z-image-turbo-controlnet",
			"controlnet_available": resp.ControlNetAvail,
			"depth_available":      resp.DepthAvail,
		})
	}
}

// RedesignHandlerV2 handles room redesign via any backend.
func RedesignHandlerV2(backend InferenceBackend) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req RedesignRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, RedesignResponse{
				Success: false,
				Error:   "Invalid request: " + err.Error(),
			})
			return
		}

		if req.Prompt == "" && req.Style == "" && req.RoomType == "" {
			c.JSON(http.StatusBadRequest, RedesignResponse{
				Success: false,
				Error:   "Provide at least one of: prompt, style, room_type",
			})
			return
		}

		// Apply defaults
		if req.Mode == "" {
			req.Mode = "auto"
		}
		if req.ControlNetType == "" {
			req.ControlNetType = "depth_v3"
		}
		if req.ControlNetStrength == 0 {
			req.ControlNetStrength = 1.0
		}
		if req.ControlNetEndStep == 0 {
			req.ControlNetEndStep = 5
		}
		if req.Strength == 0 {
			req.Strength = 0.75
		}
		if req.Steps == 0 {
			req.Steps = 9
		}
		if req.Seed == 0 {
			req.Seed = -1
		}
		if req.MaxDim == 0 {
			req.MaxDim = 1024
		}

		startTime := time.Now()
		pyReq := PythonRequest{
			Action:             "redesign",
			Image:              req.Image,
			Prompt:             req.Prompt,
			Style:              req.Style,
			RoomType:           req.RoomType,
			Mode:               req.Mode,
			ControlNetType:     req.ControlNetType,
			ControlNetStrength: req.ControlNetStrength,
			ControlNetEndStep:  req.ControlNetEndStep,
			Strength:           req.Strength,
			Steps:              req.Steps,
			GuidanceScale:      req.GuidanceScale,
			Seed:               req.Seed,
			MaxDim:             req.MaxDim,
		}

		resp, err := backend.SendRequest(pyReq)
		if err != nil {
			c.JSON(http.StatusInternalServerError, RedesignResponse{
				Success: false,
				Error:   "Inference failed: " + err.Error(),
			})
			return
		}

		if !resp.Success {
			c.JSON(http.StatusInternalServerError, RedesignResponse{
				Success: false,
				Error:   resp.Error,
			})
			return
		}

		totalTime := time.Since(startTime).Milliseconds()

		c.JSON(http.StatusOK, RedesignResponse{
			Success:            true,
			Image:              resp.Image,
			Width:              resp.Width,
			Height:             resp.Height,
			GenerationTimeMs:   totalTime,
			Seed:               resp.Seed,
			PromptUsed:         resp.PromptUsed,
			Strength:           resp.Strength,
			Device:             resp.Device,
			ControlMapUsed:     resp.ControlMapUsed,
			Mode:               resp.Mode,
			ControlNetEndStep:  resp.ControlNetEndStep,
			ControlNetStrength: resp.ControlNetStrength,
		})
	}
}

// StylesHandlerV2 returns available styles via any backend.
func StylesHandlerV2(backend InferenceBackend) gin.HandlerFunc {
	return func(c *gin.Context) {
		resp, err := backend.SendRequest(PythonRequest{Action: "styles"})
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Failed to get styles: " + err.Error(),
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"styles":     resp.Styles,
			"room_types": resp.RoomTypes,
		})
	}
}

// ControlMapsHandlerV2 returns control map info via any backend.
func ControlMapsHandlerV2(backend InferenceBackend) gin.HandlerFunc {
	return func(c *gin.Context) {
		resp, err := backend.SendRequest(PythonRequest{Action: "control_maps"})
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Failed to get control maps: " + err.Error(),
			})
			return
		}

		c.JSON(http.StatusOK, resp)
	}
}
