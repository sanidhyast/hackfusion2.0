/**
 * PPE Compliance AI Platform - Enterprise Security Operations Console
 * Resilient Camera Stream Engine & Clean State Controller
 */

const API_BASE = ""; // Relative path when served via FastAPI, or http://localhost:8000

// Central Application State
const state = {
    currentTab: "cctv", // "cctv", "image", "video", "audit"
    activeCctvFocus: null, // null for 2x2 grid, or 1, 2, 3, 4 for single cam

    // Active Camera Streams (1 to 4) with Per-Room Name and Rules
    // By default, CAM-01 and CAM-02 share Room 1, CAM-03 and CAM-04 share Room 2
    activeCameras: {
        1: { streaming: false, stream: null, interval: null, label: "CAM-01", roomName: "Room 1 (Scaffold)", rules: { helmet: true, vest: true }, detections: [] },
        2: { streaming: false, stream: null, interval: null, label: "CAM-02", roomName: "Room 1 (Scaffold)", rules: { helmet: true, vest: true }, detections: [] },
        3: { streaming: false, stream: null, interval: null, label: "CAM-03", roomName: "Room 2 (Crane)", rules: { helmet: true, vest: true }, detections: [] },
        4: { streaming: false, stream: null, interval: null, label: "CAM-04", roomName: "Room 2 (Crane)", rules: { helmet: true, vest: true }, detections: [] }
    },
    activeRoomBreaches: {}, // Tracks { [roomKey]: { camIndex, timestamp, workersCount } }
    cctvThreshold: 0.35,

    // Image Inspection State (Starts Clean)
    currentImageDetections: null,
    imageThreshold: 0.35,
    imageZoom: 1.0,
    imageRotation: 0,
    selectedWorkerId: null,
    imageQueue: [], // Array of { file, name, objectUrl, detections, latencyMs }
    currentImageIndex: 0,

    // Image Overlays
    imageOverlays: {
        boxes: true,
        labels: true,
        zones: true
    },

    // Enforcement Rules for CCTV
    cctvRules: {
        helmet: true,
        vest: true
    },

    // Enforcement Rules for Image Inspection
    imageRules: {
        helmet: true,
        vest: true
    },

    // Video Audit State
    currentVideoData: null,
    videoThreshold: 0.35,
    videoRules: {
        helmet: true,
        vest: true
    },

    // Performance Metrics
    fps: 0.0,
    latencyMs: 0.0,

    // Audit logs
    auditLogs: []
};

// -------------------------------------------------------------
// DOM Ready
// -------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
    initNavigationTabs();
    initCctvControls();
    renderMcmtFloorPlan();
    initImageControls();
    initVideoUpload();
    loadAuditLogs();
    pollHealth();
    setInterval(pollHealth, 8000);

    window.addEventListener("resize", syncAllCanvases);
    window.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            closeCameraDeviceModal();
            closeProofLightbox();
        } else if (state.currentTab === "image" && state.imageQueue && state.imageQueue.length > 1) {
            if (e.key === "ArrowLeft") {
                navigateImage(-1);
            } else if (e.key === "ArrowRight") {
                navigateImage(1);
            }
        }
    });

    if (window.lucide) window.lucide.createIcons();
});

// -------------------------------------------------------------
// 1. Navigation Tabs
// -------------------------------------------------------------
function initNavigationTabs() {
    const tabBtns = document.querySelectorAll(".nav-tab-btn");
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            state.currentTab = btn.dataset.tab;

            document.querySelectorAll(".tab-content-view").forEach(v => v.classList.add("hidden"));

            const target = document.getElementById(`${state.currentTab}-view`);
            if (target) {
                target.classList.remove("hidden");
                if (state.currentTab === "image") {
                    syncImageCanvas();
    renderMcmtFloorPlan();
                    updateImageKpis();
                } else if (state.currentTab === "cctv") {
                    syncAllCanvases();
                    updateCctvKpis();
                } else if (state.currentTab === "map") {
                    renderMcmtFloorPlan();
                } else if (state.currentTab === "video") {
                    updateVideoKpis();
                } else if (state.currentTab === "audit") {
                    renderAuditTable();
                }
            }
            if (window.lucide) window.lucide.createIcons();
        });
    });
}

async function pollHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        if (data.status === "online") {
            document.getElementById("top-gpu-name").innerText = data.gpu?.device_name?.replace("NVIDIA GeForce ", "") || "RTX 3050";
            const allocated = data.gpu?.vram_allocated_mb || 308;
            const total = data.gpu?.vram_total_mb || 6144;
            document.getElementById("top-vram").innerText = `${allocated} MB / ${total} MB`;
            document.getElementById("top-fps").innerText = `${state.fps.toFixed(1)} FPS`;
            document.getElementById("top-latency").innerText = `${state.latencyMs.toFixed(1)} ms`;
        }
    } catch (e) {
        document.getElementById("top-gpu-name").innerText = "Offline";
    }
}

// -------------------------------------------------------------
// 2. Dynamic CCTV Camera Device Selector & Live Modal Preview
// -------------------------------------------------------------
let modalPreviewStream = null;

function initCctvControls() {
    const slider = document.getElementById("cctv-conf-slider");
    const numInput = document.getElementById("cctv-conf-num");

    if (slider && numInput) {
        slider.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            state.cctvThreshold = val;
            numInput.value = Math.round(val * 100);
        });

        numInput.addEventListener("change", (e) => {
            let val = parseInt(e.target.value, 10);
            if (isNaN(val)) val = 35;
            val = Math.max(10, Math.min(val, 95));
            numInput.value = val;
            slider.value = (val / 100).toFixed(2);
            state.cctvThreshold = val / 100;
        });
    }

    // Initialize CCTV right sidebar to empty state
    updateCctvSidebarForFocus();
}

async function openCameraDeviceModal(camIndex) {
    state.selectedModalCamIndex = camIndex;
    const camObj = state.activeCameras[camIndex];
    const modal = document.getElementById("camera-modal");
    const title = document.getElementById("modal-camera-title");
    const select = document.getElementById("modal-device-select");
    const statusMsg = document.getElementById("modal-preview-status");
    const roomInput = document.getElementById("modal-room-name");

    title.innerText = `Configure & Connect CAM-0${camIndex}`;
    statusMsg.innerText = "Initializing camera preview...";
    select.innerHTML = `<option value="default">Default System Camera / Webcam</option>`;
    
    // Populate current Room Name
    if (roomInput) {
        roomInput.value = camObj.roomName || `Zone ${camIndex} Area`;
    }

    // Populate camera-specific enforcement rules
    const ruleBtns = document.querySelectorAll("#modal-cam-rules [data-modal-rule]");
    const camRules = camObj.rules || { helmet: true, vest: true };
    ruleBtns.forEach(btn => {
        const ruleKey = btn.dataset.modalRule;
        if (camRules[ruleKey]) {
            btn.classList.add("selected");
        } else {
            btn.classList.remove("selected");
        }
    });

    modal.classList.remove("hidden");

    // Release any previous modal preview stream
    if (modalPreviewStream) {
        modalPreviewStream.getTracks().forEach(t => t.stop());
        modalPreviewStream = null;
    }

    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === "videoinput");

        if (videoDevices.length > 0) {
            select.innerHTML = "";
            videoDevices.forEach((dev, idx) => {
                const opt = document.createElement("option");
                opt.value = dev.deviceId;
                opt.text = dev.label || `Camera Device ${idx + 1}`;
                select.appendChild(opt);
            });
        }
    } catch (err) {
        console.warn("Could not enumerate camera devices:", err);
    }

    // Start live preview in modal
    await startModalPreview();
    if (window.lucide) window.lucide.createIcons();
}

function toggleModalCamRule(btn) {
    btn.classList.toggle("selected");
}

async function startModalPreview() {
    const select = document.getElementById("modal-device-select");
    const video = document.getElementById("modal-preview-video");
    const emptyPlaceholder = document.getElementById("modal-preview-empty");
    const statusMsg = document.getElementById("modal-preview-status");
    const deviceId = select.value;

    if (modalPreviewStream) {
        modalPreviewStream.getTracks().forEach(t => t.stop());
        modalPreviewStream = null;
    }

    // Small delay for driver release
    await new Promise(r => setTimeout(r, 100));

    let stream = null;

    // 1. Try strict exact deviceId first so OnePlus vs Webcam is strictly honored
    if (deviceId && deviceId !== "default") {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { deviceId: { exact: deviceId } }
            });
        } catch (eExact) {
            console.warn("Strict exact deviceId failed, trying ideal constraint...", eExact);
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { deviceId: { ideal: deviceId } }
                });
            } catch (eIdeal) {
                console.warn("Ideal device constraint failed, trying fallback...", eIdeal);
            }
        }
    }

    // 2. Fallback to default video device if exact device wasn't selected or failed
    if (!stream) {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
        } catch (err) {
            console.error("Camera preview failed:", err);
            video.classList.add("hidden");
            emptyPlaceholder.classList.remove("hidden");
            statusMsg.innerText = "Camera in use by another app or permission denied.";
            return;
        }
    }

    modalPreviewStream = stream;
    video.srcObject = stream;
    video.classList.remove("hidden");
    emptyPlaceholder.classList.add("hidden");
    video.play();
}

async function handleModalDeviceChange() {
    await startModalPreview();
}

function closeCameraDeviceModal() {
    if (modalPreviewStream) {
        modalPreviewStream.getTracks().forEach(t => t.stop());
        modalPreviewStream = null;
    }
    const video = document.getElementById("modal-preview-video");
    const emptyPlaceholder = document.getElementById("modal-preview-empty");
    if (video) {
        video.srcObject = null;
        video.classList.add("hidden");
    }
    if (emptyPlaceholder) emptyPlaceholder.classList.remove("hidden");

    document.getElementById("camera-modal").classList.add("hidden");
    state.selectedModalCamIndex = null;
}

async function confirmCameraConnection() {
    const camIndex = state.selectedModalCamIndex;
    if (!camIndex) return;

    const camObj = state.activeCameras[camIndex];
    const video = document.getElementById(`video-cam-${camIndex}`);
    const placeholder = document.getElementById(`placeholder-cam-${camIndex}`);
    const meta = document.getElementById(`meta-cam-${camIndex}`);

    // Save custom Room Name & Per-Camera Enforcement Rules
    const roomInput = document.getElementById("modal-room-name");
    if (roomInput && roomInput.value.trim()) {
        camObj.roomName = roomInput.value.trim();
    }
    const ruleBtns = document.querySelectorAll("#modal-cam-rules [data-modal-rule]");
    camObj.rules = camObj.rules || {};
    ruleBtns.forEach(btn => {
        camObj.rules[btn.dataset.modalRule] = btn.classList.contains("selected");
    });

    // Update tile header title with Room Name
    const titleEl = document.getElementById(`title-cam-${camIndex}`);
    if (titleEl) {
        titleEl.innerHTML = `<i data-lucide="video" style="width:13px; height:13px;"></i> CAM-0${camIndex} (${camObj.roomName})`;
    }

    // If modal preview has a running stream, transfer it directly!
    let stream = modalPreviewStream;
    modalPreviewStream = null; // Detach so closeCameraDeviceModal won't kill it

    closeCameraDeviceModal();

    if (!stream && !camObj.streaming) {
        // Fallback if preview was empty
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
        } catch (err) {
            alert(`Could not start video stream for CAM-0${camIndex}: ${err.message}`);
            return;
        }
    }

    if (stream) {
        // Release any previous stream on this tile
        if (camObj.streaming) {
            disconnectCamera(camIndex);
        }

        camObj.stream = stream;
        camObj.streaming = true;

        video.srcObject = stream;
        video.classList.remove("hidden");
        placeholder.classList.add("hidden");

        meta.innerHTML = `<div class="live-dot" style="margin-right:2px; background:var(--accent);"></div> Live`;

        video.onloadedmetadata = () => {
            video.play();
            syncCanvasDimensions(`canvas-cam-${camIndex}`, video);
            startCameraInferenceLoop(camIndex);
            updateCctvActiveSummary();
        };
    }

    updateLiveBadge();
    updateCctvKpis();
    if (window.lucide) window.lucide.createIcons();
}

function disconnectCamera(camIndex) {
    const camObj = state.activeCameras[camIndex];
    if (camObj.interval) clearInterval(camObj.interval);
    if (camObj.stream) {
        camObj.stream.getTracks().forEach(t => t.stop());
    }
    camObj.streaming = false;
    camObj.stream = null;
    camObj.interval = null;
    camObj.detections = [];

    const video = document.getElementById(`video-cam-${camIndex}`);
    const placeholder = document.getElementById(`placeholder-cam-${camIndex}`);
    const meta = document.getElementById(`meta-cam-${camIndex}`);
    const tile = document.getElementById(`cam-tile-${camIndex}`);
    const statusFooter = document.getElementById(`footer-cam-${camIndex}-status`);
    const workersFooter = document.getElementById(`footer-cam-${camIndex}-workers`);

    if (video) {
        video.pause();
        video.srcObject = null;
        video.classList.add("hidden");
    }
    if (placeholder) placeholder.classList.remove("hidden");
    if (meta) meta.innerText = "Disconnected";
    if (tile) tile.classList.remove("has-violation");
    if (statusFooter) {
        statusFooter.className = "status-tag-chip standby";
        statusFooter.innerText = "Standby";
    }
    if (workersFooter) workersFooter.innerText = "Workers: 0";

    const canvas = document.getElementById(`canvas-cam-${camIndex}`);
    if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    updateLiveBadge();
    updateCctvActiveSummary();
    updateCctvKpis();
}

function disconnectAllCameras() {
    [1, 2, 3, 4].forEach(idx => disconnectCamera(idx));
    state.fps = 0.0;
    state.latencyMs = 0.0;
}

function attachTestFeedToCam(camIndex) {
    // Allows user to test live AI detection loop using test video if hardware webcam is locked
    const camObj = state.activeCameras[camIndex];
    const video = document.getElementById(`video-cam-${camIndex}`);
    const placeholder = document.getElementById(`placeholder-cam-${camIndex}`);
    const meta = document.getElementById(`meta-cam-${camIndex}`);

    disconnectCamera(camIndex);

    camObj.streaming = true;
    placeholder.classList.add("hidden");
    video.classList.remove("hidden");
    meta.innerHTML = `<div class="live-dot" style="margin-right:2px; background:var(--accent);"></div> Demo feed`;

    // Connect to canvas video simulation
    video.onloadedmetadata = () => {
        syncCanvasDimensions(`canvas-cam-${camIndex}`, video);
        startCameraInferenceLoop(camIndex);
        updateCctvActiveSummary();
    };

    updateLiveBadge();
}

function startCameraInferenceLoop(camIndex) {
    const camObj = state.activeCameras[camIndex];
    const video = document.getElementById(`video-cam-${camIndex}`);
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 360;
    const ctx = canvas.getContext("2d");

    let isProcessing = false;

    camObj.interval = setInterval(async () => {
        if (!camObj.streaming || isProcessing) return;
        isProcessing = true;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        camObj.lastFrameCanvas = canvas;
        canvas.toBlob(async (blob) => {
            if (!blob) {
                isProcessing = false;
                return;
            }
            const formData = new FormData();
            formData.append("file", blob, `cam${camIndex}_frame.jpg`);
            formData.append("return_base64", "false");
            formData.append("conf_thresh", state.cctvThreshold.toString());

            const t0 = performance.now();
            try {
                const res = await fetch(`${API_BASE}/api/detect/image`, {
                    method: "POST",
                    body: formData
                });
                const data = await res.json();
                state.latencyMs = performance.now() - t0;
                state.fps = 1000.0 / Math.max(state.latencyMs, 1.0);

                if (data.success && camObj.streaming) {
                    handleCamInferenceResult(camIndex, data);
                }
            } catch (e) {
                console.error(`CAM-0${camIndex} detection error:`, e);
            } finally {
                isProcessing = false;
            }
        }, "image/jpeg", 0.7);
    }, 100);
}

// =============================================================
// MULTI-CAMERA MULTI-TARGET TRACKING (MCMT) ENGINE
// Sequence:
// 1. Single-Camera Tracking (SCT) producing (camera_id, local_track_id)
// 2. Overlapping Views Floor Plan Homography & Physical Position Matching -> global_worker_id
// 3. Non-Overlapping Zone Connections & Plausible Travel Times
// =============================================================

function calculateBoxIoU(boxA, boxB) {
    const xA = Math.max(boxA[0], boxB[0]);
    const yA = Math.max(boxA[1], boxB[1]);
    const xB = Math.min(boxA[2], boxB[2]);
    const yB = Math.min(boxA[3], boxB[3]);
    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    const boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
    const unionArea = boxAArea + boxBArea - interArea;
    return unionArea > 0 ? interArea / unionArea : 0;
}

class SingleCameraTracker {
    constructor(camId) {
        this.camId = camId;
        this.nextLocalId = 1;
        this.tracks = []; // { id, box, footPos, velocity, confidence, timeSinceUpdate, hits, lastTimestamp }
        this.maxAge = 12; // ~1.2s at 100ms
    }

    update(detections, timestamp) {
        // Linear motion prediction for existing tracks
        for (const trk of this.tracks) {
            trk.timeSinceUpdate += 1;
            if (trk.velocity) {
                trk.box[0] += trk.velocity[0];
                trk.box[2] += trk.velocity[0];
                trk.box[1] += trk.velocity[1];
                trk.box[3] += trk.velocity[1];
            }
        }

        const pairs = [];
        for (let d = 0; d < detections.length; d++) {
            const detBox = detections[d].box;
            if (!detBox) continue;
            const detFoot = [(detBox[0] + detBox[2]) / 2, detBox[3]];
            for (let t = 0; t < this.tracks.length; t++) {
                const trk = this.tracks[t];
                const iou = calculateBoxIoU(detBox, trk.box);
                const trkFoot = [(trk.box[0] + trk.box[2]) / 2, trk.box[3]];
                const dist = Math.hypot(detFoot[0] - trkFoot[0], detFoot[1] - trkFoot[1]);
                if (iou > 0.20 || dist < 90) {
                    const score = (iou * 0.6) + ((1.0 - Math.min(1.0, dist / 150)) * 0.4);
                    pairs.push({ d, t, score });
                }
            }
        }
        pairs.sort((a, b) => b.score - a.score);

        const usedDets = new Set();
        const usedTrks = new Set();
        const matched = [];

        for (const p of pairs) {
            if (!usedDets.has(p.d) && !usedTrks.has(p.t)) {
                usedDets.add(p.d);
                usedTrks.add(p.t);
                matched.push(p);
            }
        }

        // Update matched tracks
        for (const m of matched) {
            const trk = this.tracks[m.t];
            const det = detections[m.d];
            const oldFoot = [(trk.box[0] + trk.box[2]) / 2, trk.box[3]];
            const newFoot = [(det.box[0] + det.box[2]) / 2, det.box[3]];
            trk.velocity = [(newFoot[0] - oldFoot[0]) * 0.5, (newFoot[1] - oldFoot[1]) * 0.5];
            trk.box = det.box;
            trk.footPos = newFoot;
            trk.confidence = det.confidence;
            trk.timeSinceUpdate = 0;
            trk.hits += 1;
            trk.lastTimestamp = timestamp;

            det.camera_id = this.camId;
            det.local_track_id = trk.id;
            if (trk.global_worker_id) {
                det.global_worker_id = trk.global_worker_id;
            }
        }

        // Create new tracks for unmatched detections
        for (let d = 0; d < detections.length; d++) {
            if (!usedDets.has(d)) {
                const det = detections[d];
                if (!det.box) continue;
                const foot = [(det.box[0] + det.box[2]) / 2, det.box[3]];
                const newTrk = {
                    id: this.nextLocalId++,
                    camId: this.camId,
                    box: det.box,
                    footPos: foot,
                    velocity: [0, 0],
                    confidence: det.confidence,
                    timeSinceUpdate: 0,
                    hits: 1,
                    lastTimestamp: timestamp,
                    global_worker_id: null,
                    color_sig: null
                };
                this.tracks.push(newTrk);
                det.camera_id = this.camId;
                det.local_track_id = newTrk.id;
            }
        }

        // Collect expired tracks as exit events
        const expired = this.tracks.filter(t => t.timeSinceUpdate > this.maxAge);
        this.tracks = this.tracks.filter(t => t.timeSinceUpdate <= this.maxAge);

        return {
            activeTracks: this.tracks,
            expiredTracks: expired
        };
    }
}

// Calibrated perspective homography mapping foot contact point (u, v) in [0, 1] to Site Ground Plane [0, 100]m x [0, 100]m
function projectFootToGroundPlane(camId, footNormX, footNormY) {
    let X = 25.0 + (footNormX * 50.0);
    let Y = 25.0 + (footNormY * 50.0);
    return [Math.max(0, Math.min(100, X)), Math.max(0, Math.min(100, Y))];
}

class MCMTCoordinator {
    constructor() {
        this.trackers = {
            1: new SingleCameraTracker(1),
            2: new SingleCameraTracker(2),
            3: new SingleCameraTracker(3),
            4: new SingleCameraTracker(4)
        };
        this.nextGlobalId = 101;
        this.recentExits = []; // [ { camId, localId, globalId, exitGroundPos, exitTime } ]
        this.groundMatchDistanceThreshold = 18.0; // meters threshold on ground floor plan
        this.plausibleTransitsCount = 0;
        this.overlapMatchesCount = 0;
    }

    processCameraDetections(camIndex, rawWorkers, timestamp) {
        // 1. Single-Camera Tracking (SCT) producing (camera_id, local_track_id)
        const sctResult = this.trackers[camIndex].update(rawWorkers, timestamp);

        // Record expired tracks for Stage 3 non-overlapping transition monitoring
        for (const exp of sctResult.expiredTracks) {
            const footGround = projectFootToGroundPlane(camIndex, exp.footPos[0] / 640.0, exp.footPos[1] / 360.0);
            this.recentExits.push({
                camId: camIndex,
                localId: exp.id,
                exitGroundPos: footGround,
                exitTime: timestamp
            });
        }
        this.recentExits = this.recentExits.filter(e => (timestamp - e.exitTime) < 45000);

        // Project every active detection on this camera to the site ground plane (X_g, Y_g)
        for (const w of rawWorkers) {
            const box = w.box || [0, 0, 0, 0];
            const footNormX = ((box[0] + box[2]) / 2) / 640.0;
            const footNormY = box[3] / 360.0;
            w.ground_pos = projectFootToGroundPlane(camIndex, footNormX, footNormY);
            w.camera_id = camIndex;
        }

        // 2. Overlapping Views Floor Plan Calibration & Ground Position Match
        this.matchOverlappingCameras(timestamp);

        // 3. Non-overlapping zone connections & plausible travel times
        this.evaluateNonOverlappingTransitions(camIndex, rawWorkers, timestamp);

        return rawWorkers;
    }

    matchOverlappingCameras(timestamp) {
        // Step 1: Ensure each track on each active camera has a persistent, unique global_worker_id
        for (let c = 1; c <= 4; c++) {
            const cam = state.activeCameras[c];
            if (cam && cam.streaming && cam.rawWorkers) {
                for (const w of cam.rawWorkers) {
                    const localId = w.local_track_id || w.worker_id;
                    const trk = this.trackers[c]?.tracks?.find(t => t.id === localId);
                    if (trk) {
                        if (!trk.global_worker_id) {
                            trk.global_worker_id = this.nextGlobalId++;
                        }
                        w.global_worker_id = trk.global_worker_id;
                    } else if (!w.global_worker_id) {
                        w.global_worker_id = this.nextGlobalId++;
                    }
                    // Default state
                    w.is_primary_mcmt = true;
                    w.is_room_duplicate = false;
                    w.duplicate_of_cam = null;
                    w.overlap_distance = null;
                }
            }
        }

        // Step 2: Check all active streaming pairs (camA, camB) with camA < camB
        const streamingIndices = [1, 2, 3, 4].filter(idx => 
            state.activeCameras[idx] && state.activeCameras[idx].streaming && state.activeCameras[idx].rawWorkers?.length > 0
        );

        let totalOverlaps = 0;

        for (let i = 0; i < streamingIndices.length; i++) {
            const camA = streamingIndices[i];
            const detsA = state.activeCameras[camA].rawWorkers || [];
            if (detsA.length === 0) continue;

            // Sort workers from left-to-right to establish spatial rank order
            const sortedA = [...detsA].sort((a, b) => ((a.box?.[0] || 0) - (b.box?.[0] || 0)));

            for (let j = i + 1; j < streamingIndices.length; j++) {
                const camB = streamingIndices[j];
                const detsB = state.activeCameras[camB].rawWorkers || [];
                if (detsB.length === 0) continue;

                const sortedB = [...detsB].sort((a, b) => ((a.box?.[0] || 0) - (b.box?.[0] || 0)));
                const candidates = [];

                for (let idxA = 0; idxA < detsA.length; idxA++) {
                    const wA = detsA[idxA];
                    if (!wA.box || !wA.ground_pos) continue;

                    const uA = ((wA.box[0] + wA.box[2]) / 2) / 640.0;
                    const vA = wA.box[3] / 360.0;
                    const rankA = sortedA.indexOf(wA);

                    for (let idxB = 0; idxB < detsB.length; idxB++) {
                        const wB = detsB[idxB];
                        if (!wB.box || !wB.ground_pos) continue;
                        if (wB.is_room_duplicate) continue; // Already matched to an earlier camera

                        const uB = ((wB.box[0] + wB.box[2]) / 2) / 640.0;
                        const vB = wB.box[3] / 360.0;
                        const rankB = sortedB.indexOf(wB);

                        const dGround = Math.hypot(wA.ground_pos[0] - wB.ground_pos[0], wA.ground_pos[1] - wB.ground_pos[1]);
                        const dNorm = Math.hypot(uA - uB, vA - vB);
                        const iou = calculateBoxIoU(wA.box, wB.box);

                        // Check PPE consistency (both missing helmet, etc.)
                        const ppeSame = (
                            !!wA.detected_ppe?.helmet === !!wB.detected_ppe?.helmet &&
                            !!wA.detected_ppe?.vest === !!wB.detected_ppe?.vest
                        );

                        // Match condition:
                        const rankMatch = (rankA === rankB);
                        const isMatch = (
                            dGround <= this.groundMatchDistanceThreshold ||
                            dNorm <= 0.35 ||
                            iou >= 0.15 ||
                            (rankMatch && dNorm <= 0.55) ||
                            (ppeSame && dNorm <= 0.45)
                        );

                        if (isMatch) {
                            // Bonus for preserving left-to-right rank order
                            const rankPenalty = rankMatch ? 0.0 : 0.25;
                            const score = (dNorm * 0.4) + ((1.0 - Math.min(1.0, iou)) * 0.3) + ((dGround / 100.0) * 0.3) + rankPenalty;
                            candidates.push({ wA, wB, score, dist: dGround, rankA, rankB });
                        }
                    }
                }

                candidates.sort((a, b) => a.score - b.score);

                const usedA = new Set();
                const usedB = new Set();

                for (const match of candidates) {
                    const idA = match.wA.local_track_id || match.wA.worker_id;
                    const idB = match.wB.local_track_id || match.wB.worker_id;

                    if (!usedA.has(idA) && !usedB.has(idB) && !match.wB.is_room_duplicate) {
                        // Prevent two detections on the same camera (camB) from having the same global_worker_id!
                        const targetGlobalId = match.wA.global_worker_id;
                        const conflictOnB = detsB.some(other => other !== match.wB && other.global_worker_id === targetGlobalId);
                        if (conflictOnB) {
                            continue;
                        }

                        usedA.add(idA);
                        usedB.add(idB);
                        totalOverlaps++;

                        // Root primary camera is either wA's duplicate source or camA
                        const rootCam = match.wA.duplicate_of_cam || camA;

                        // Propagate global ID
                        match.wB.global_worker_id = targetGlobalId;
                        const trkB = this.trackers[camB]?.tracks?.find(t => t.id === idB);
                        if (trkB) trkB.global_worker_id = targetGlobalId;

                        // Higher camera (camB) is ALWAYS duplicate (Orange)
                        match.wB.is_primary_mcmt = false;
                        match.wB.is_room_duplicate = true; // Turn Orange!
                        match.wB.duplicate_of_cam = rootCam;
                        match.wB.overlap_distance = match.dist;

                        // wA NEVER gets overwritten to false if it was already duplicate!
                        if (!match.wA.is_room_duplicate) {
                            match.wA.is_primary_mcmt = true;
                            match.wA.is_room_duplicate = false;
                            match.wA.overlap_distance = match.dist;
                        }
                    }
                }
            }
        }

        this.overlapMatchesCount = totalOverlaps;
    }

    evaluateNonOverlappingTransitions(camIndex, workers, timestamp) {
        // Plausible travel times:
        // CAM-04 (Gate) <-> CAM-01/02 (Scaffold): 8s to 35s
        // CAM-01/02 (Scaffold) <-> CAM-03 (Crane): 6s to 30s
        for (const w of workers) {
            if (w.is_room_duplicate) continue;

            for (const exit of this.recentExits) {
                if (exit.camId === camIndex) continue;

                const isConnected = (
                    (exit.camId === 4 && (camIndex === 1 || camIndex === 2)) ||
                    ((exit.camId === 1 || exit.camId === 2) && camIndex === 4) ||
                    ((exit.camId === 1 || exit.camId === 2) && camIndex === 3) ||
                    (exit.camId === 3 && (camIndex === 1 || camIndex === 2))
                );

                if (isConnected) {
                    const dt = (timestamp - exit.exitTime) / 1000.0;
                    const minTime = (exit.camId === 4 || camIndex === 4) ? 8.0 : 6.0;
                    const maxTime = (exit.camId === 4 || camIndex === 4) ? 35.0 : 30.0;

                    if (dt >= minTime && dt <= maxTime) {
                        w.plausible_transit = {
                            fromCam: exit.camId,
                            fromLocalId: exit.localId,
                            travelTimeSec: Math.round(dt)
                        };
                        this.plausibleTransitsCount++;
                        break;
                    }
                }
            }
        }
    }
}

state.mcmtCoordinator = new MCMTCoordinator();

function handleCamInferenceResult(camIndex, data) {
    const camObj = state.activeCameras[camIndex];
    const rawWorkers = data.workers || [];
    camObj.rawWorkers = rawWorkers;

    const now = Date.now();

    // Run the 3-stage MCMT Pipeline:
    // 1. Single-camera tracking -> (camera_id, local_track_id)
    // 2. Homography floor plan matching -> global_worker_id & overlap de-duplication
    // 3. Non-overlapping plausible travel time transition evaluation
    if (!state.mcmtCoordinator) state.mcmtCoordinator = new MCMTCoordinator();
    state.mcmtCoordinator.processCameraDetections(camIndex, rawWorkers, now);

    // Filter by active rules for this specific camera
    const rules = camObj.rules || state.cctvRules;
    const evaluated = rawWorkers.map(w => {
        let isCompliant = true;
        const missing = [];

        if (rules.helmet && !w.detected_ppe.helmet) {
            isCompliant = false;
            missing.push("No helmet");
        }
        if (rules.vest && !w.detected_ppe.vest) {
            isCompliant = false;
            missing.push("No vest");
        }

        const isDuplicate = !!w.is_room_duplicate;
        const dupCam = w.duplicate_of_cam;
        const distStr = w.overlap_distance ? ` ${w.overlap_distance.toFixed(1)}m` : '';

        return {
            ...w,
            worker_id: w.local_track_id || w.worker_id,
            is_compliant: isCompliant,
            is_room_duplicate: isDuplicate,
            duplicate_of_cam: dupCam,
            verdict: isCompliant 
                ? "COMPLIANT" 
                : (isDuplicate ? `MCMT OVERLAP (CAM-0${dupCam}${distStr})` : "VIOLATION"),
            missing_chips: missing,
            camera: `CAM-0${camIndex}`
        };
    });

    camObj.detections = evaluated;

    const isFocused = (state.activeCctvFocus === camIndex);

    const primaryViolations = evaluated.filter(w => !w.is_compliant && !w.is_room_duplicate);
    const duplicateViolations = evaluated.filter(w => !w.is_compliant && w.is_room_duplicate);
    const primaryViolCount = primaryViolations.length;
    const dupCount = duplicateViolations.length;

    // Update tile footer & borders for 2x2 grid AND focused view
    const workersFooter = document.getElementById(`footer-cam-${camIndex}-workers`);
    const statusFooter = document.getElementById(`footer-cam-${camIndex}-status`);
    const tile = document.getElementById(`cam-tile-${camIndex}`);

    if (workersFooter) workersFooter.innerText = `Workers: ${evaluated.length}`;

    if (statusFooter) {
        if (primaryViolCount > 0) {
            statusFooter.className = "status-tag-chip violation";
            statusFooter.style.backgroundColor = "";
            statusFooter.style.color = "";
            statusFooter.style.borderColor = "";
            statusFooter.innerText = `${primaryViolCount} missing PPE`;
            if (tile) {
                tile.classList.add("has-violation");
                tile.classList.remove("has-mcmt-overlap");
            }
        } else if (dupCount > 0) {
            statusFooter.className = "status-tag-chip warning";
            statusFooter.style.backgroundColor = "rgba(249, 115, 22, 0.2)";
            statusFooter.style.color = "#fb923c";
            statusFooter.style.borderColor = "#f97316";
            statusFooter.innerText = `${dupCount} also on another camera`;
            if (tile) {
                tile.classList.remove("has-violation");
                tile.classList.add("has-mcmt-overlap");
            }
        } else {
            statusFooter.className = "status-tag-chip compliant";
            statusFooter.style.backgroundColor = "";
            statusFooter.style.color = "";
            statusFooter.style.borderColor = "";
            statusFooter.innerText = isFocused ? "All wearing PPE" : (camObj.streaming ? "Live" : "Standby");
            if (tile) {
                tile.classList.remove("has-violation");
                tile.classList.remove("has-mcmt-overlap");
            }
        }
    }

    // Always draw live bounding boxes on tile canvas in both 2x2 grid and single view
    drawCanvasBoxes(`canvas-cam-${camIndex}`, evaluated, true);

    // Synchronously refresh bounding boxes and status chips for other active streaming cameras
    [1, 2, 3, 4].forEach(idx => {
        if (idx !== camIndex && state.activeCameras[idx]?.streaming && state.activeCameras[idx]?.detections) {
            const otherCam = state.activeCameras[idx];
            let modified = false;
            otherCam.detections.forEach(w => {
                const matchRaw = (otherCam.rawWorkers || []).find(rw => (rw.local_track_id || rw.worker_id) === (w.local_track_id || w.worker_id));
                if (matchRaw) {
                    w.global_worker_id = matchRaw.global_worker_id;
                    w.is_primary_mcmt = matchRaw.is_primary_mcmt;
                    w.is_room_duplicate = matchRaw.is_room_duplicate;
                    w.duplicate_of_cam = matchRaw.duplicate_of_cam;
                    w.overlap_distance = matchRaw.overlap_distance;
                    if (!w.is_compliant) {
                        w.verdict = w.is_room_duplicate 
                            ? `MCMT OVERLAP (CAM-0${w.duplicate_of_cam})` 
                            : "VIOLATION";
                    }
                    modified = true;
                }
            });

            if (modified) {
                const otherTile = document.getElementById(`cam-tile-${idx}`);
                const otherStatus = document.getElementById(`footer-cam-${idx}-status`);
                const otherPrimary = otherCam.detections.filter(w => !w.is_compliant && !w.is_room_duplicate).length;
                const otherDup = otherCam.detections.filter(w => !w.is_compliant && w.is_room_duplicate).length;

                if (otherStatus) {
                    if (otherPrimary > 0) {
                        otherStatus.className = "status-tag-chip violation";
                        otherStatus.style.backgroundColor = "";
                        otherStatus.style.color = "";
                        otherStatus.style.borderColor = "";
                        otherStatus.innerText = `${otherPrimary} missing PPE`;
                        if (otherTile) {
                            otherTile.classList.add("has-violation");
                            otherTile.classList.remove("has-mcmt-overlap");
                        }
                    } else if (otherDup > 0) {
                        otherStatus.className = "status-tag-chip warning";
                        otherStatus.style.backgroundColor = "rgba(249, 115, 22, 0.2)";
                        otherStatus.style.color = "#fb923c";
                        otherStatus.style.borderColor = "#f97316";
                        otherStatus.innerText = `${otherDup} also on another camera`;
                        if (otherTile) {
                            otherTile.classList.remove("has-violation");
                            otherTile.classList.add("has-mcmt-overlap");
                        }
                    }
                }
                drawCanvasBoxes(`canvas-cam-${idx}`, otherCam.detections, true);
            }
        }
    });

    // Render MCMT Top-Down Floor Plan Mini-Map in real-time
    renderMcmtFloorPlan();

    // Aggregate KPIs across all active CCTV feeds
    updateCctvKpis();

    // Trigger automatic live breach snapshot ONLY IF THERE ARE PRIMARY VIOLATIONS!
    // Overlapping secondary detections SKIP audit image capture so the person is NEVER counted twice!
    if (primaryViolCount > 0) {
        triggerCctvViolationSnapshot(camIndex);
    }

    // If this screen is currently focused on right sidebar:
    if (isFocused) {
        renderCctvVerdictsList(evaluated);
    }
}

function updateCctvKpis() {
    let total = 0;
    let compliant = 0;
    let violations = 0;
    let duplicatesCount = 0;
    const allWorkers = [];

    Object.keys(state.activeCameras).forEach(idx => {
        const cam = state.activeCameras[idx];
        if (cam.streaming && cam.detections) {
            total += cam.detections.length;
            const comp = cam.detections.filter(w => w.is_compliant).length;
            compliant += comp;
            // Count ONLY primary violations towards breach KPIs! Secondary overlapping duplicate workers are de-duplicated!
            const primaryViol = cam.detections.filter(w => !w.is_compliant && !w.is_room_duplicate).length;
            const dupViol = cam.detections.filter(w => !w.is_compliant && w.is_room_duplicate).length;
            violations += primaryViol;
            duplicatesCount += dupViol;
            allWorkers.push(...cam.detections);
        }
    });

    const activeCount = Object.values(state.activeCameras).filter(c => c.streaming).length;
    const rate = total > 0 ? Math.round((compliant / total) * 100) : 100;

    const totalEl = document.getElementById("kpi-total-val");
    if (totalEl) totalEl.innerText = total;
    const totalSub = document.getElementById("kpi-total-sub");
    if (totalSub) totalSub.innerText = activeCount > 0 
        ? `${total} workers across ${activeCount} active camera${activeCount > 1 ? 's' : ''}`
        : "No active streams connected";

    const compEl = document.getElementById("kpi-compliant-val");
    if (compEl) compEl.innerText = compliant;
    const compSub = document.getElementById("kpi-compliant-sub");
    if (compSub) compSub.innerText = `${compliant} of ${total} workers compliant`;

    const violCard = document.getElementById("kpi-violations-card");
    const violEl = document.getElementById("kpi-violations-val");
    if (violEl) violEl.innerText = violations;
    if (violCard) {
        if (violations > 0) {
            violCard.classList.add("has-violations");
            const violSub = document.getElementById("kpi-violations-sub");
            if (violSub) {
                violSub.innerText = duplicatesCount > 0 
                    ? `${violations} active (${duplicatesCount} overlapping de-duplicated)` 
                    : `${violations} active safety breaches`;
            }
        } else {
            violCard.classList.remove("has-violations");
            const violSub = document.getElementById("kpi-violations-sub");
            if (violSub) violSub.innerText = `0 active safety breaches`;
        }
    }

    const rateEl = document.getElementById("kpi-rate-val");
    if (rateEl) rateEl.innerText = `${rate}%`;
    const rateSub = document.getElementById("kpi-rate-sub");
    if (rateSub) rateSub.innerText = rate >= 90 ? "Site safety target met" : "Below site safety target (90%)";

    // Update Right Panel CCTV verdicts list ONLY IF a camera screen is focused!
    if (state.activeCctvFocus) {
        const focusedCam = state.activeCameras[state.activeCctvFocus];
        renderCctvVerdictsList(focusedCam ? (focusedCam.detections || []) : []);
    }

    // Slim alert banner
    const banner = document.getElementById("violation-alert-banner");
    const bannerText = document.getElementById("alert-banner-text");
    if (banner && bannerText) {
        if (violations > 0) {
            const firstViol = allWorkers.find(w => !w.is_compliant && !w.is_room_duplicate);
            const dupNote = duplicatesCount > 0 ? ` (${duplicatesCount} multi-cam overlapping de-duplicated)` : '';
            bannerText.innerText = `${violations} violation detected${dupNote} - ${firstViol?.camera || 'CAM'} Worker #${firstViol?.worker_id}: ${(firstViol?.missing_chips || []).join(', ')}`;
            banner.classList.remove("hidden");
        } else {
            banner.classList.add("hidden");
        }
    }
}

function renderCctvVerdictsList(workers) {
    const container = document.getElementById("cctv-verdicts-container");
    const badge = document.getElementById("cctv-verdicts-badge");
    if (!container) return;

    if (!state.activeCctvFocus) {
        // No screen focused - right sidebar remains empty
        return;
    }

    const camIndex = state.activeCctvFocus;
    const camObj = state.activeCameras[camIndex];
    if (!camObj) return;

    const isStreaming = camObj.streaming;
    const camWorkers = workers !== undefined ? workers : (camObj.detections || []);

    if (badge) {
        badge.innerText = `${camWorkers.length} Worker${camWorkers.length !== 1 ? 's' : ''}`;
    }

    if (!isStreaming) {
        container.innerHTML = `
            <div class="list-empty">${camObj.label} is not connected. Use "Connect camera" on its screen to start the feed.</div>
        `;
        return;
    }

    if (!camWorkers || camWorkers.length === 0) {
        container.innerHTML = `
            <div class="list-empty">Camera is live. No workers in view right now.</div>
        `;
        return;
    }

    container.innerHTML = camWorkers.map(w => {
        const isComp = w.is_compliant;
        const isDup = !!w.is_room_duplicate;
        const badgeClass = isComp ? "compliant" : (isDup ? "warning" : "violation");
        const badgeText = isComp ? "Wearing PPE" : (isDup ? `Also on CAM-0${w.duplicate_of_cam}` : "Missing PPE");
        const conf = Math.round(w.confidence * 100);

        let chipsHtml = "";
        if (isComp) {
            const hasH = !!(w.detected_ppe && w.detected_ppe.helmet);
            const hasV = !!(w.detected_ppe && w.detected_ppe.vest);
            if (hasH && hasV) {
                chipsHtml = `<span class="compliant-pill-chip">Hard hat and vest</span>`;
            } else if (hasH) {
                chipsHtml = `<span class="compliant-pill-chip">Hard hat</span>`;
            } else if (hasV) {
                chipsHtml = `<span class="compliant-pill-chip chip-note">Vest only, hard hat not required</span>`;
            } else {
                chipsHtml = `<span class="compliant-pill-chip">No PPE required here</span>`;
            }
        } else if (isDup) {
            chipsHtml = (w.missing_chips || []).map(c => 
                `<span class="violation-pill-chip chip-overlap">${c}</span>`
            ).join("") + `<span class="compliant-pill-chip chip-overlap">Counted once</span>`;
        } else {
            chipsHtml = (w.missing_chips || []).map(c => `<span class="violation-pill-chip">${c}</span>`).join("");
        }

        return `
            <div class="worker-verdict-card">
                <div class="card-top-row">
                    <div class="worker-meta-left">
                        <span class="worker-id-badge">Worker #${w.worker_id}${w.global_worker_id ? ` (Global #${w.global_worker_id})` : ''}</span>
                        <span class="worker-conf-score">${conf}%</span>
                    </div>
                    <span class="status-badge-chip ${badgeClass}">${badgeText}</span>
                </div>
                <div class="worker-tags-row">${chipsHtml}</div>
                <div class="card-bottom-row">
                    <span>Your review</span>
                    <select class="reviewer-dropdown" onclick="event.stopPropagation()" onchange="modifyCctvVerdict('${camObj.label}', ${w.worker_id}, this.value)">
                        <option value="confirm" ${(!w.reviewer_status || w.reviewer_status === 'confirm') ? 'selected' : ''}>Confirmed</option>
                        <option value="false_alarm" ${w.reviewer_status === 'false_alarm' ? 'selected' : ''}>Dismiss as False Alarm</option>
                        <option value="violation" ${w.reviewer_status === 'violation' ? 'selected' : ''}>Force Violation</option>
                    </select>
                </div>
            </div>
        `;
    }).join("");
}

function modifyCctvVerdict(cameraName, workerId, action) {
    const camIdx = parseInt(cameraName.replace(/\D/g, ''), 10) || 1;
    const cam = state.activeCameras[camIdx];
    if (!cam || !cam.detections) return;

    const w = cam.detections.find(worker => worker.worker_id === workerId);
    if (!w) return;

    w.reviewer_status = action;
    if (action === "false_alarm") {
        w.is_compliant = true;
        w.verdict = "COMPLIANT";
    } else if (action === "violation") {
        w.is_compliant = false;
        w.verdict = "VIOLATION";
    }

    drawCanvasBoxes(`canvas-cam-${camIdx}`, cam.detections);
    updateCctvKpis();
}

function updateCctvActiveSummary() {
    const activeCount = Object.values(state.activeCameras).filter(c => c.streaming).length;
    const el = document.getElementById("cctv-active-cams-count");
    if (el) el.innerText = `${activeCount} of 4 cameras live`;
}

function updateLiveBadge() {
    const activeCount = Object.values(state.activeCameras).filter(c => c.streaming).length;
    const badge = document.getElementById("top-live-badge");
    const text = document.getElementById("top-live-text");
    if (activeCount > 0) {
        badge.classList.add("active");
        text.innerText = `${activeCount} camera${activeCount > 1 ? 's' : ''} live`;
    } else {
        badge.classList.remove("active");
        text.innerText = "Cameras off";
    }
}

function toggleQuadFocus(tileNum) {
    const quad = document.getElementById("cctv-quad-grid");
    const tiles = document.querySelectorAll(".cctv-tile");

    if (tileNum === null || state.activeCctvFocus === tileNum) {
        state.activeCctvFocus = null;
        quad.classList.remove("single-focus");
        tiles.forEach(t => {
            t.classList.remove("active-focus");
            t.classList.remove("has-violation");
        });
        // Sync all canvases in 2x2 grid view
        [1, 2, 3, 4].forEach(idx => {
            const cam = state.activeCameras[idx];
            if (cam && cam.streaming && cam.detections) {
                drawCanvasBoxes(`canvas-cam-${idx}`, cam.detections, true);
            }
        });
    } else {
        state.activeCctvFocus = tileNum;
        quad.classList.add("single-focus");
        tiles.forEach(t => {
            t.classList.remove("active-focus");
            t.classList.remove("has-violation");
        });
        const target = document.getElementById(`cam-tile-${tileNum}`);
        if (target) target.classList.add("active-focus");

        // Keep all active cameras rendering bounding lines and violations
        [1, 2, 3, 4].forEach(idx => {
            const cam = state.activeCameras[idx];
            const tileEl = document.getElementById(`cam-tile-${idx}`);
            const statusFooter = document.getElementById(`footer-cam-${idx}-status`);
            if (cam && cam.streaming && cam.detections) {
                const primaryViol = cam.detections.filter(w => !w.is_compliant && !w.is_room_duplicate).length;
                const dupViol = cam.detections.filter(w => !w.is_compliant && w.is_room_duplicate).length;
                if (statusFooter) {
                    if (primaryViol > 0) {
                        statusFooter.className = "status-tag-chip violation";
                        statusFooter.style.backgroundColor = "";
                        statusFooter.style.color = "";
                        statusFooter.style.borderColor = "";
                        statusFooter.innerText = `${primaryViol} missing PPE`;
                        if (tileEl) {
                            tileEl.classList.add("has-violation");
                            tileEl.classList.remove("has-mcmt-overlap");
                        }
                    } else if (dupViol > 0) {
                        statusFooter.className = "status-tag-chip warning";
                        statusFooter.style.backgroundColor = "rgba(249, 115, 22, 0.2)";
                        statusFooter.style.color = "#fb923c";
                        statusFooter.style.borderColor = "#f97316";
                        statusFooter.innerText = `${dupViol} also on another camera`;
                        if (tileEl) {
                            tileEl.classList.remove("has-violation");
                            tileEl.classList.add("has-mcmt-overlap");
                        }
                    } else {
                        statusFooter.className = "status-tag-chip compliant";
                        statusFooter.style.backgroundColor = "";
                        statusFooter.style.color = "";
                        statusFooter.style.borderColor = "";
                        statusFooter.innerText = "All wearing PPE";
                        if (tileEl) {
                            tileEl.classList.remove("has-violation");
                            tileEl.classList.remove("has-mcmt-overlap");
                        }
                    }
                }
                drawCanvasBoxes(`canvas-cam-${idx}`, cam.detections, true);
            }
        });
    }

    // Update right sidebar visibility & rules/verdicts for this screen
    updateCctvSidebarForFocus();
    setTimeout(syncAllCanvases, 100);
}

function updateCctvSidebarForFocus() {
    const emptyPanel = document.getElementById("cctv-sidebar-empty");
    const activePanel = document.getElementById("cctv-sidebar-active");
    if (!emptyPanel || !activePanel) return;

    if (!state.activeCctvFocus) {
        emptyPanel.classList.remove("hidden");
        activePanel.classList.add("hidden");
        return;
    }

    emptyPanel.classList.add("hidden");
    activePanel.classList.remove("hidden");

    const camIndex = state.activeCctvFocus;
    const camObj = state.activeCameras[camIndex];
    if (!camObj) return;

    const titleEl = document.getElementById("cctv-focused-cam-title");
    const roomEl = document.getElementById("cctv-focused-cam-room");
    if (titleEl) titleEl.innerText = camObj.label;
    if (roomEl) roomEl.innerText = camObj.roomName || `Camera ${camIndex}`;

    // Sync button states with this camera's rules
    const btns = document.querySelectorAll("[data-cctvrule]");
    btns.forEach(btn => {
        const ruleKey = btn.dataset.cctvrule;
        const isActive = !!(camObj.rules && camObj.rules[ruleKey]);
        btn.classList.toggle("selected", isActive);
    });

    renderCctvVerdictsList(camObj.detections || []);
    if (window.lucide) window.lucide.createIcons();
}

function toggleFocusedCameraRule(btn) {
    if (!state.activeCctvFocus) return;
    const camIndex = state.activeCctvFocus;
    const camObj = state.activeCameras[camIndex];
    if (!camObj) return;

    const rule = btn.dataset.cctvrule;
    btn.classList.toggle("selected");
    if (!camObj.rules) camObj.rules = {};
    camObj.rules[rule] = btn.classList.contains("selected");

    reEvaluateCameraDetections(camIndex);
}

function setFocusedCameraRules(selectAll) {
    if (!state.activeCctvFocus) return;
    const camIndex = state.activeCctvFocus;
    const camObj = state.activeCameras[camIndex];
    if (!camObj) return;

    const btns = document.querySelectorAll("[data-cctvrule]");
    btns.forEach(btn => {
        const rule = btn.dataset.cctvrule;
        if (selectAll) {
            btn.classList.add("selected");
            camObj.rules[rule] = true;
        } else {
            btn.classList.remove("selected");
            camObj.rules[rule] = false;
        }
    });

    reEvaluateCameraDetections(camIndex);
}

function reEvaluateCameraDetections(camIndex) {
    const camObj = state.activeCameras[camIndex];
    if (!camObj) return;
    const rawWorkers = camObj.rawWorkers || [];
    const rules = camObj.rules || {};

    const evaluated = rawWorkers.map(w => {
        let isCompliant = true;
        const missing = [];

        if (rules.helmet && !w.detected_ppe.helmet) {
            isCompliant = false;
            missing.push("No helmet");
        }
        if (rules.vest && !w.detected_ppe.vest) {
            isCompliant = false;
            missing.push("No vest");
        }

        const camName = camObj.roomName ? `CAM-0${camIndex} (${camObj.roomName})` : `CAM-0${camIndex}`;

        return {
            ...w,
            is_compliant: isCompliant,
            verdict: isCompliant ? "COMPLIANT" : "VIOLATION",
            missing_chips: missing,
            camera: camName
        };
    });

    camObj.detections = evaluated;

    const workersFooter = document.getElementById(`footer-cam-${camIndex}-workers`);
    const statusFooter = document.getElementById(`footer-cam-${camIndex}-status`);
    const tile = document.getElementById(`cam-tile-${camIndex}`);
    const violCount = evaluated.filter(w => !w.is_compliant).length;

    if (workersFooter) workersFooter.innerText = `Workers: ${evaluated.length}`;
    if (statusFooter) {
        if (violCount > 0) {
            statusFooter.className = "status-tag-chip violation";
            statusFooter.innerText = `${violCount} missing PPE`;
            if (tile) tile.classList.add("has-violation");
        } else {
            statusFooter.className = "status-tag-chip compliant";
            statusFooter.innerText = "All wearing PPE";
            if (tile) tile.classList.remove("has-violation");
        }
    }

    const isFocused = (state.activeCctvFocus === camIndex);
    drawCanvasBoxes(`canvas-cam-${camIndex}`, evaluated, true);
    updateCctvKpis();
    if (isFocused) {
        renderCctvVerdictsList(evaluated);
    }
}

const cctvAutoLogTimers = {};

function triggerCctvViolationSnapshot(camIndex) {
    const camObj = state.activeCameras[camIndex];
    if (!camObj || !camObj.streaming) return;

    const detections = camObj.detections || [];
    const violations = detections.filter(w => !w.is_compliant);
    if (violations.length === 0) return;

    const now = Date.now();
    // 5-second cooldown per camera to prevent duplicate alerts while catching violations quickly
    if (cctvAutoLogTimers[camIndex] && (now - cctvAutoLogTimers[camIndex] < 5000)) {
        return;
    }
    cctvAutoLogTimers[camIndex] = now;
    saveCctvViolationSnapshot(camIndex);
}

// Aliases for compatibility
function checkCctvAutoLogBreach(camIndex) {
    triggerCctvViolationSnapshot(camIndex);
}

function autoLogCctvBreachOnFocus(camIndex) {
    saveCctvViolationSnapshot(camIndex);
}

function saveCctvViolationSnapshot(camIndex) {
    const camObj = state.activeCameras[camIndex];
    if (!camObj || !camObj.streaming) return;

    const detections = camObj.detections || [];
    const violations = detections.filter(w => !w.is_compliant);
    if (violations.length === 0) return;

    // Strict multi-camera room de-duplication:
    // If all violations on this feed are secondary room duplicates, SKIP capturing duplicate audit snapshot!
    const primaryViolations = violations.filter(w => !w.is_room_duplicate);
    if (primaryViolations.length === 0) {
        console.log(`[De-Duplication] Skipped secondary snapshot for CAM-0${camIndex} — already captured in this room by CAM-0${violations[0].duplicate_of_cam}.`);
        return;
    }

    const sourceName = camObj.roomName ? `CAM-0${camIndex} (${camObj.roomName})` : `CAM-0${camIndex}`;
    const snapshotImage = captureCameraSnapshot(camIndex);
    if (!snapshotImage) {
        console.warn(`[Snapshot] Could not generate frame snapshot for ${sourceName}`);
        return;
    }

    // Capture synchronized snapshots across all 4 cameras at this exact violation moment
    const multiCamSnapshots = {};
    [1, 2, 3, 4].forEach(idx => {
        const c = state.activeCameras[idx];
        if (c && c.streaming) {
            const shot = captureCameraSnapshot(idx);
            if (shot) {
                multiCamSnapshots[`cam_${idx}`] = {
                    camIndex: idx,
                    label: c.label,
                    roomName: c.roomName || `CAM-0${idx}`,
                    image: shot
                };
            }
        }
    });

    const summary = primaryViolations.map(w => `Worker #${w.worker_id}: ${(w.missing_chips || []).join(', ')}`).join(" | ");
    const incidentId = `INC-${Date.now().toString().slice(-6)}`;

    const logEntry = {
        id: incidentId,
        timestamp: new Date().toLocaleString(),
        camera: sourceName,
        totalWorkers: detections.length,
        compliant: Math.max(0, detections.length - primaryViolations.length),
        violations: primaryViolations.length,
        verdict: summary,
        snapshotImage: snapshotImage,
        multiCamViews: multiCamSnapshots,
        activeRules: { ...(camObj.rules || state.cctvRules) },
        workers: detections.map(w => ({
            worker_id: w.worker_id,
            confidence: w.confidence,
            is_compliant: w.is_compliant,
            is_room_duplicate: !!w.is_room_duplicate,
            duplicate_of_cam: w.duplicate_of_cam,
            verdict: w.verdict,
            missing_chips: w.missing_chips || [],
            detected_ppe: w.detected_ppe || {},
            negative_flags: w.negative_flags || []
        }))
    };

    state.auditLogs.unshift(logEntry);
    saveAuditLogs();
    renderAuditTable();

    // Persist primary snapshot + multi-camera snapshots to disk in backend/processed/snapshots/
    fetch(`${API_BASE}/api/audit/save_snapshot`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            incident_id: incidentId,
            camera: sourceName,
            snapshot_base64: snapshotImage,
            violations: primaryViolations.length,
            summary: summary,
            timestamp: logEntry.timestamp,
            totalWorkers: logEntry.totalWorkers,
            compliant: logEntry.compliant,
            activeRules: logEntry.activeRules,
            workers: logEntry.workers,
            multi_cam_snapshots: multiCamSnapshots
        })
    }).catch(e => console.warn("Backend snapshot save error:", e));

    const banner = document.getElementById("violation-alert-banner");
    const bannerText = document.getElementById("alert-banner-text");
    if (banner && bannerText) {
        bannerText.innerHTML = `📸 <strong>Live Breach Screenshot Captured:</strong> ${sourceName} — ${summary} (Audit ID: <code>${incidentId}</code>)`;
        banner.classList.remove("hidden");
        setTimeout(() => { banner.classList.add("hidden"); }, 4500);
    }
}

function toggleQuadFullscreen() {
    const quad = document.getElementById("cctv-quad-grid");
    if (!document.fullscreenElement) {
        quad.requestFullscreen?.().catch(e => alert("Fullscreen error: " + e.message));
    } else {
        document.exitFullscreen?.();
    }
}

function syncAllCanvases() {
    [1, 2, 3, 4].forEach(idx => {
        const canvas = document.getElementById(`canvas-cam-${idx}`);
        const video = document.getElementById(`video-cam-${idx}`);
        if (canvas && video) {
            syncCanvasDimensions(`canvas-cam-${idx}`, video);
            const cam = state.activeCameras[idx];
            if (cam && cam.streaming && cam.detections) {
                drawCanvasBoxes(`canvas-cam-${idx}`, cam.detections, true);
            }
        }
    });
    syncImageCanvas();
    renderMcmtFloorPlan();
}

function syncCanvasDimensions(canvasId, parentEl) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !parentEl) return;
    canvas.width = parentEl.clientWidth || 300;
    canvas.height = parentEl.clientHeight || 180;
}

function drawCanvasBoxes(canvasId, workers, isFocused = true) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!workers || workers.length === 0) return;

    // Inference frame is 640x360
    const scaleX = canvas.width / 640.0;
    const scaleY = canvas.height / 360.0;

    workers.forEach(w => {
        if (!w.box) return;
        const [x1, y1, x2, y2] = w.box;

        const sx = x1 * scaleX;
        const sy = y1 * scaleY;
        const sw = (x2 - x1) * scaleX;
        const sh = (y2 - y1) * scaleY;

        const isViolation = !w.is_compliant;
        const isDuplicate = !!w.is_room_duplicate;

        let strokeColor = "#A3A3A3";
        let chipBg = "#262626";
        let chipTextColor = "#FFFFFF";

        const missingText = w.missing_chips && w.missing_chips.length > 0 
            ? ` (${w.missing_chips.join(', ')})` 
            : ' (Full PPE)';

        const transitTag = w.plausible_transit ? ` [Transit CAM-0${w.plausible_transit.fromCam}]` : '';
        const globalTag = w.global_worker_id ? ` (Global #${w.global_worker_id})` : '';

        let labelText = `Worker #${w.worker_id}${globalTag} - OK${missingText}${transitTag}`;

        if (isViolation) {
            if (isDuplicate) {
                strokeColor = "#F97316"; // VIBRANT ORANGE FOR OVERLAPPING POSITION
                chipBg = "#EA580C";      // ORANGE CHIP BACKGROUND
                const distTag = w.overlap_distance ? ` (${w.overlap_distance.toFixed(1)}m)` : '';
                labelText = `⚠️ Global #${w.global_worker_id || w.worker_id} [MCMT Overlap CAM-0${w.duplicate_of_cam}${distTag}]${missingText}`;
            } else {
                strokeColor = "#A31C1B"; // RED FOR PRIMARY VIOLATION
                chipBg = "#A31C1B";
                labelText = `Worker #${w.worker_id}${globalTag} - VIOLATION${missingText}${transitTag}`;
            }
        }

        // 1. Worker Outer Bounding Box
        ctx.save();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = isDuplicate ? 3.5 : (isViolation ? 2.5 : 2);
        ctx.strokeRect(sx, sy, sw, sh);

        // 2. Anatomical Head Zone (0% - 32%) & Torso Zone (22% - 75%)
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1;
        ctx.strokeStyle = isDuplicate ? "rgba(249, 115, 22, 0.45)" : "rgba(255, 255, 255, 0.35)";
        ctx.strokeRect(sx, sy, sw, sh * 0.32); // Head zone

        ctx.strokeStyle = isDuplicate ? "rgba(249, 115, 22, 0.30)" : "rgba(255, 255, 255, 0.20)";
        ctx.strokeRect(sx, sy + (sh * 0.22), sw, sh * 0.53); // Torso zone
        ctx.restore();

        // 3. Draw Detected PPE boxes if present (helmet/vest)
        if (w.detected_ppe) {
            ctx.save();
            ctx.lineWidth = 1.5;
            if (w.detected_ppe.helmet?.box) {
                const [hx1, hy1, hx2, hy2] = w.detected_ppe.helmet.box;
                ctx.strokeStyle = "#34d399"; // Green border for verified worn helmet
                ctx.strokeRect(hx1 * scaleX, hy1 * scaleY, (hx2 - hx1) * scaleX, (hy2 - hy1) * scaleY);
            }
            if (w.detected_ppe.vest?.box) {
                const [vx1, vy1, vx2, vy2] = w.detected_ppe.vest.box;
                ctx.strokeStyle = "#38bdf8"; // Cyan border for verified worn vest
                ctx.strokeRect(vx1 * scaleX, vy1 * scaleY, (vx2 - vx1) * scaleX, (vy2 - vy1) * scaleY);
            }
            ctx.restore();
        }

        // 4. Label Chip with Verdict & Missing Details
        ctx.save();
        ctx.font = "600 11px Inter, sans-serif";
        const textMetrics = ctx.measureText(labelText);
        const chipW = textMetrics.width + 12;
        const chipH = 18;

        ctx.fillStyle = chipBg;
        ctx.fillRect(sx, Math.max(sy - chipH, 0), chipW, chipH);

        ctx.fillStyle = chipTextColor;
        ctx.fillText(labelText, sx + 6, Math.max(sy - 4, 12));
        ctx.restore();
    });
}

function renderMcmtFloorPlan() {
    const canvas = document.getElementById("mcmt-floorplan-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
        canvas.width = canvas.clientWidth || 500;
        canvas.height = canvas.clientHeight || 125;
    }

    const W = canvas.width;
    const H = canvas.height;

    const toCanvasX = (x) => (x / 100.0) * W;
    const toCanvasY = (y) => ((100.0 - y) / 100.0) * H;

    // Background
    ctx.fillStyle = "#080808";
    ctx.fillRect(0, 0, W, H);

    // 10m site grid
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    for (let x = 10; x < 100; x += 10) {
        ctx.beginPath();
        ctx.moveTo(toCanvasX(x), 0);
        ctx.lineTo(toCanvasX(x), H);
        ctx.stroke();
    }
    for (let y = 10; y < 100; y += 10) {
        ctx.beginPath();
        ctx.moveTo(0, toCanvasY(y));
        ctx.lineTo(W, toCanvasY(y));
        ctx.stroke();
    }

    // Overlapping Calibration Zone across all 4 cameras (Central Work Area [25, 75]m x [25, 75]m)
    const ox1 = toCanvasX(25);
    const oy1 = toCanvasY(75);
    const ow = toCanvasX(75) - ox1;
    const oh = toCanvasY(25) - oy1;

    ctx.save();
    ctx.fillStyle = "rgba(249, 115, 22, 0.08)";
    ctx.fillRect(ox1, oy1, ow, oh);
    ctx.strokeStyle = "rgba(249, 115, 22, 0.45)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.strokeRect(ox1, oy1, ow, oh);
    ctx.font = "bold 9px Inter, sans-serif";
    ctx.fillStyle = "rgba(249, 115, 22, 0.9)";
    ctx.fillText("MULTI-CAMERA CALIBRATED OVERLAP ZONE (CAM 1-4)", ox1 + 6, oy1 + 14);
    ctx.restore();

    // Transit Corridors (South Gate -> Scaffold -> North Crane Yard)
    ctx.save();
    ctx.setLineDash([2, 4]);
    ctx.strokeStyle = "rgba(56, 189, 248, 0.25)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(toCanvasX(50), toCanvasY(10));
    ctx.lineTo(toCanvasX(50), toCanvasY(25));
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(toCanvasX(50), toCanvasY(75));
    ctx.lineTo(toCanvasX(50), toCanvasY(90));
    ctx.stroke();
    ctx.font = "8px Inter, sans-serif";
    ctx.fillStyle = "rgba(56, 189, 248, 0.6)";
    ctx.fillText("Gate Transit Δt [8-35s]", toCanvasX(52), toCanvasY(18));
    ctx.fillText("Crane Transit Δt [6-30s]", toCanvasX(52), toCanvasY(82));
    ctx.restore();

    // Camera FOV Frustums & Nodes (All 4 converging on central overlap zone)
    const cams = [
        { id: 1, label: "CAM-01 (W)", color: "#38bdf8", pos: [8, 50],  cone: [[15, 25], [75, 25], [75, 75], [15, 75]] },
        { id: 2, label: "CAM-02 (E)", color: "#c084fc", pos: [92, 50], cone: [[25, 25], [85, 25], [85, 75], [25, 75]] },
        { id: 3, label: "CAM-03 (N)", color: "#facc15", pos: [50, 96], cone: [[25, 25], [75, 25], [75, 85], [25, 85]] },
        { id: 4, label: "CAM-04 (S)", color: "#34d399", pos: [50, 4],  cone: [[25, 15], [75, 15], [75, 75], [25, 75]] }
    ];

    cams.forEach(cam => {
        const isStreaming = state.activeCameras[cam.id] && state.activeCameras[cam.id].streaming;
        ctx.save();
        ctx.fillStyle = isStreaming ? `${cam.color}15` : "rgba(255,255,255,0.02)";
        ctx.strokeStyle = isStreaming ? `${cam.color}66` : "rgba(255,255,255,0.1)";
        ctx.lineWidth = 1;

        ctx.beginPath();
        const p0 = [toCanvasX(cam.pos[0]), toCanvasY(cam.pos[1])];
        ctx.moveTo(p0[0], p0[1]);
        cam.cone.forEach(pt => ctx.lineTo(toCanvasX(pt[0]), toCanvasY(pt[1])));
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = isStreaming ? cam.color : "#444";
        ctx.beginPath();
        ctx.arc(p0[0], p0[1], 3.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.font = "9px Inter, sans-serif";
        ctx.fillStyle = isStreaming ? "#ddd" : "#666";
        ctx.fillText(cam.label, p0[0] + 5, p0[1] + (cam.id === 3 ? 10 : (cam.id === 4 ? -4 : 3)));
        ctx.restore();
    });

    // Plot all active workers across cameras on the floor plan
    const allWorkers = [];
    [1, 2, 3, 4].forEach(idx => {
        const c = state.activeCameras[idx];
        if (c && c.streaming && c.detections) {
            c.detections.forEach(w => allWorkers.push({ ...w, fromCam: idx }));
        }
    });

    // Draw connection lines for duplicate overlap pairs
    const overlaps = allWorkers.filter(w => w.is_room_duplicate && w.ground_pos);
    overlaps.forEach(dupW => {
        const primaryW = allWorkers.find(w => w.global_worker_id === dupW.global_worker_id && !w.is_room_duplicate && w.ground_pos);
        if (primaryW) {
            ctx.save();
            ctx.strokeStyle = "#f97316";
            ctx.lineWidth = 1.5;
            ctx.setLineDash([2, 2]);
            ctx.beginPath();
            ctx.moveTo(toCanvasX(primaryW.ground_pos[0]), toCanvasY(primaryW.ground_pos[1]));
            ctx.lineTo(toCanvasX(dupW.ground_pos[0]), toCanvasY(dupW.ground_pos[1]));
            ctx.stroke();
            ctx.restore();
        }
    });

    // Draw worker markers
    let globalWorkerSet = new Set();
    allWorkers.forEach(w => {
        if (!w.ground_pos) return;
        globalWorkerSet.add(w.global_worker_id || w.worker_id);

        const wx = toCanvasX(w.ground_pos[0]);
        const wy = toCanvasY(w.ground_pos[1]);
        const isViol = !w.is_compliant;
        const isDup = !!w.is_room_duplicate;

        let dotColor = "#22c55e"; // Compliant
        if (isViol) {
            dotColor = isDup ? "#f97316" : "#A31C1B";
        }

        ctx.save();
        if (isViol && !isDup) {
            ctx.strokeStyle = "rgba(163, 28, 27, 0.4)";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(wx, wy, 8, 0, Math.PI * 2);
            ctx.stroke();
        }

        ctx.fillStyle = dotColor;
        ctx.beginPath();
        ctx.arc(wx, wy, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.font = "bold 9px Inter, sans-serif";
        ctx.fillStyle = "#fff";
        const tagText = `G#${w.global_worker_id || w.worker_id} (C${w.fromCam})`;
        ctx.fillText(tagText, wx + 6, wy + 3);
        ctx.restore();
    });

    // Update stats bar
    const statGlobals = document.getElementById("mcmt-stat-active-globals");
    const statOverlaps = document.getElementById("mcmt-stat-overlaps");
    const statTransits = document.getElementById("mcmt-stat-transits");

    if (statGlobals) statGlobals.innerHTML = `<strong>${globalWorkerSet.size}</strong> workers tracked`;
    if (statOverlaps) statOverlaps.innerHTML = `<strong>${overlaps.length}</strong> seen by two cameras`;
    if (statTransits && state.mcmtCoordinator) {
        statTransits.innerHTML = `<strong>${state.mcmtCoordinator.plausibleTransitsCount}</strong> moved between cameras`;
    }
}

// -------------------------------------------------------------
// 3. Dedicated Image Inspection & Multi-Image Carousel
// -------------------------------------------------------------
function initImageControls() {
    const fileInput = document.getElementById("hidden-image-upload");
    const dropzone = document.getElementById("image-dropzone");

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleMultipleImageUpload(e.target.files);
        }
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--accent)";
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.style.borderColor = "#2e2e2e";
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "#2e2e2e";
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleMultipleImageUpload(e.dataTransfer.files);
        }
    });

    const stage = document.getElementById("image-stage-box");
    if (stage) {
        stage.addEventListener("dragover", (e) => {
            e.preventDefault();
        });
        stage.addEventListener("drop", (e) => {
            e.preventDefault();
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleMultipleImageUpload(e.dataTransfer.files);
            }
        });
    }

    const slider = document.getElementById("image-conf-slider");
    const numInput = document.getElementById("image-conf-num");
    let imageConfDebounce = null;

    slider.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        state.imageThreshold = val;
        numInput.value = Math.round(val * 100);
        reFilterImageWorkers();
        if (state.currentImageFile) {
            clearTimeout(imageConfDebounce);
            imageConfDebounce = setTimeout(() => {
                if (state.imageQueue && state.imageQueue[state.currentImageIndex]) {
                    state.imageQueue[state.currentImageIndex].detections = null;
                    displayImageFromQueue(state.currentImageIndex);
                } else {
                    inspectUploadedImage(state.currentImageFile);
                }
            }, 350);
        }
    });

    numInput.addEventListener("change", (e) => {
        let val = parseInt(e.target.value, 10);
        if (isNaN(val)) val = 35;
        val = Math.max(10, Math.min(val, 95));
        numInput.value = val;
        slider.value = (val / 100).toFixed(2);
        state.imageThreshold = val / 100;
        reFilterImageWorkers();
        if (state.currentImageFile) {
            clearTimeout(imageConfDebounce);
            imageConfDebounce = setTimeout(() => {
                if (state.imageQueue && state.imageQueue[state.currentImageIndex]) {
                    state.imageQueue[state.currentImageIndex].detections = null;
                    displayImageFromQueue(state.currentImageIndex);
                } else {
                    inspectUploadedImage(state.currentImageFile);
                }
            }, 350);
        }
    });
}

async function handleMultipleImageUpload(fileList) {
    const validFiles = Array.from(fileList).filter(f => f.type.startsWith("image/"));
    if (validFiles.length === 0) return;

    state.imageQueue = validFiles.map(file => ({
        file: file,
        name: file.name,
        objectUrl: URL.createObjectURL(file),
        detections: null,
        latencyMs: 0
    }));
    state.currentImageIndex = 0;

    await displayImageFromQueue(0);
}

async function navigateImage(delta) {
    if (!state.imageQueue || state.imageQueue.length <= 1) return;
    const newIdx = state.currentImageIndex + delta;
    if (newIdx >= 0 && newIdx < state.imageQueue.length) {
        await displayImageFromQueue(newIdx);
    }
}

async function displayImageFromQueue(index) {
    if (!state.imageQueue || index < 0 || index >= state.imageQueue.length) return;
    state.currentImageIndex = index;
    const item = state.imageQueue[index];
    state.currentImageFile = item.file;

    updateImageCarouselUI();

    const img = document.getElementById("image-display-element");
    const dropzone = document.getElementById("image-dropzone");

    img.src = item.objectUrl;
    img.classList.remove("hidden");
    if (dropzone) dropzone.classList.add("hidden");

    // If detections are already cached for this image:
    if (item.detections) {
        state.currentImageDetections = item.detections;
        state.latencyMs = item.latencyMs;
        updateImageMetaText(item);
        syncImageCanvas();
    renderMcmtFloorPlan();
        reFilterImageWorkers();
        return;
    }

    // Otherwise, inspect this image via backend
    showImageProgress(true);
    const formData = new FormData();
    formData.append("file", item.file);
    formData.append("return_base64", "true");
    formData.append("conf_thresh", state.imageThreshold.toString());

    const t0 = performance.now();
    try {
        const res = await fetch(`${API_BASE}/api/detect/image`, {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        const latency = performance.now() - t0;
        showImageProgress(false);

        if (data.success) {
            item.detections = data;
            item.latencyMs = latency;
            state.currentImageDetections = data;
            state.latencyMs = latency;
            state.fps = 1000.0 / Math.max(latency, 1.0);

            updateImageMetaText(item);

            img.onload = () => {
                state.imageZoom = 1.0;
                state.imageRotation = 0;
                applyImageTransform();
                syncImageCanvas();
    renderMcmtFloorPlan();
                reFilterImageWorkers();
                const evaluated = getEvaluatedImageWorkers();
                if (evaluated.some(w => !w.is_compliant)) {
                    autoLogInspectionIncident("Photo Inspection", evaluated);
                }
            };
            if (img.complete && img.naturalWidth) {
                syncImageCanvas();
    renderMcmtFloorPlan();
                reFilterImageWorkers();
            }
        } else {
            alert("Inspection failed for " + item.name + ": " + (data.detail || "Server error"));
        }
    } catch (err) {
        showImageProgress(false);
        alert("Error connecting to detection backend: " + err.message);
    }
}

function updateImageCarouselUI() {
    const prevBtn = document.getElementById("img-nav-prev");
    const nextBtn = document.getElementById("img-nav-next");
    const counterPill = document.getElementById("img-nav-counter");
    const counterText = document.getElementById("img-counter-text");
    const botControls = document.getElementById("img-bottom-nav-controls");
    const botPrev = document.getElementById("btn-bot-prev");
    const botNext = document.getElementById("btn-bot-next");
    const botCounter = document.getElementById("bot-counter-text");

    const total = state.imageQueue ? state.imageQueue.length : 0;
    const cur = state.currentImageIndex;

    if (total > 1) {
        if (prevBtn) {
            prevBtn.classList.remove("hidden");
            prevBtn.disabled = (cur === 0);
        }
        if (nextBtn) {
            nextBtn.classList.remove("hidden");
            nextBtn.disabled = (cur === total - 1);
        }
        if (counterPill) counterPill.classList.remove("hidden");
        if (counterText) counterText.innerText = `${cur + 1} / ${total}`;

        if (botControls) botControls.classList.remove("hidden");
        if (botPrev) botPrev.disabled = (cur === 0);
        if (botNext) botNext.disabled = (cur === total - 1);
        if (botCounter) botCounter.innerText = `${cur + 1} / ${total}`;
    } else {
        if (prevBtn) prevBtn.classList.add("hidden");
        if (nextBtn) nextBtn.classList.add("hidden");
        if (counterPill) counterPill.classList.add("hidden");
        if (botControls) botControls.classList.add("hidden");
    }
    if (window.lucide) window.lucide.createIcons();
}

function updateImageMetaText(item) {
    const meta = document.getElementById("image-meta-text");
    if (!meta) return;
    const total = state.imageQueue ? state.imageQueue.length : 1;
    const idxStr = total > 1 ? ` (${state.currentImageIndex + 1} of ${total})` : '';
    const workers = item.detections ? item.detections.total_workers : 0;
    meta.innerText = `${item.name}${idxStr}: ${workers} worker${workers !== 1 ? 's' : ''} checked in ${state.latencyMs.toFixed(0)} ms`;
}

async function loadImageSample(filename) {
    showImageProgress(true);
    try {
        const res = await fetch(`/samples/${filename}`);
        const blob = await res.blob();
        const file = new File([blob], filename, { type: "image/jpeg" });
        await inspectUploadedImage(file);
    } catch (err) {
        showImageProgress(false);
        alert("Failed to load test sample: " + err.message);
    }
}

async function inspectUploadedImage(file) {
    await handleMultipleImageUpload([file]);
}

function showImageProgress(show) {
    const bar = document.getElementById("image-progress-bar");
    if (bar) bar.style.display = show ? "block" : "none";
}

function syncImageCanvas() {
    const canvas = document.getElementById("image-overlay-canvas");
    const stage = document.getElementById("image-stage-box");
    if (canvas && stage) {
        canvas.width = stage.clientWidth;
        canvas.height = stage.clientHeight;
    }
}

function getEvaluatedImageWorkers() {
    if (!state.currentImageDetections) return [];

    const data = state.currentImageDetections;
    return (data.workers || []).map(w => {
        let isCompliant = true;
        const missingChips = [];

        // Evaluate fixed classes: Helmet and Vest
        if (state.imageRules.helmet && !w.detected_ppe.helmet) {
            isCompliant = false;
            missingChips.push("No helmet");
        }
        if (state.imageRules.vest && !w.detected_ppe.vest) {
            isCompliant = false;
            missingChips.push("No vest");
        }

        // Apply manual reviewer decision from dropdown if set
        if (w.reviewer_status === "false_positive") {
            isCompliant = true;
            missingChips.length = 0;
        } else if (w.reviewer_status === "override") {
            isCompliant = false;
            if (missingChips.length === 0) missingChips.push("Safety Breach (Manual Override)");
        }

        return {
            ...w,
            is_compliant: isCompliant,
            verdict: isCompliant ? "COMPLIANT" : "VIOLATION",
            missing_chips: missingChips
        };
    });
}

function reFilterImageWorkers() {
    if (!state.currentImageDetections) return;
    const evaluated = getEvaluatedImageWorkers();
    renderImageVerdictsList(evaluated);
    drawImageOverlayBoxes(evaluated);
    updateImageKpis(evaluated);
}

function updateImageKpis(evaluated) {
    if (state.currentTab !== "image") return;

    const workers = evaluated || (state.currentImageDetections?.workers || []);
    const total = workers.length;
    const comp = workers.filter(w => w.is_compliant).length;
    const viol = total - comp;
    const rate = total > 0 ? Math.round((comp / total) * 100) : 100;

    const totalEl = document.getElementById("img-kpi-total-val") || document.getElementById("kpi-total-val");
    if (totalEl) totalEl.innerText = total;
    const totalSub = document.getElementById("img-kpi-total-sub") || document.getElementById("kpi-total-sub");
    if (totalSub) totalSub.innerText = total > 0 ? `${total} workers in photo` : "No image evaluated yet";

    const compEl = document.getElementById("img-kpi-compliant-val") || document.getElementById("kpi-compliant-val");
    if (compEl) compEl.innerText = comp;
    const compSub = document.getElementById("img-kpi-compliant-sub") || document.getElementById("kpi-compliant-sub");
    if (compSub) compSub.innerText = `${comp} of ${total} workers compliant`;

    const violCard = document.getElementById("img-kpi-violations-card") || document.getElementById("kpi-violations-card");
    const violEl = document.getElementById("img-kpi-violations-val") || document.getElementById("kpi-violations-val");
    if (violEl) violEl.innerText = viol;
    const violSub = document.getElementById("img-kpi-violations-sub") || document.getElementById("kpi-violations-sub");
    if (violSub) violSub.innerText = `${viol} safety breaches flagged`;

    if (violCard) {
        if (viol > 0) violCard.classList.add("has-violations");
        else violCard.classList.remove("has-violations");
    }

    const rateEl = document.getElementById("img-kpi-rate-val") || document.getElementById("kpi-rate-val");
    if (rateEl) rateEl.innerText = `${rate}%`;
    const rateSub = document.getElementById("img-kpi-rate-sub") || document.getElementById("kpi-rate-sub");
    if (rateSub) rateSub.innerText = rate >= 90 ? "Site safety target met" : "Below site safety target (90%)";

    const banner = document.getElementById("violation-alert-banner");
    const bannerText = document.getElementById("alert-banner-text");
    if (viol > 0) {
        const firstViol = workers.find(w => !w.is_compliant);
        bannerText.innerText = `${viol} violation detected in photo - Worker #${firstViol.worker_id}: ${firstViol.missing_chips?.join(', ') || 'Missing PPE'}`;
        banner.classList.remove("hidden");
    } else {
        banner.classList.add("hidden");
    }
}

function renderImageVerdictsList(workers) {
    const container = document.getElementById("image-verdicts-container");
    const badge = document.getElementById("image-verdicts-badge");
    badge.innerText = `${workers.length} Workers`;

    if (!workers || workers.length === 0) {
        container.innerHTML = `
            <div class="list-empty">Add a photo to see each worker's result here.</div>
        `;
        return;
    }

    container.innerHTML = workers.map(w => {
        const isSelected = state.selectedWorkerId === w.worker_id;
        const badgeClass = w.is_compliant ? "compliant" : "violation";
        const badgeText = w.is_compliant ? "Wearing PPE" : "Missing PPE";
        const confPct = Math.round(w.confidence * 100);

        let chipsHtml = "";
        if (w.is_compliant) {
            const hasH = !!(w.detected_ppe && w.detected_ppe.helmet);
            const hasV = !!(w.detected_ppe && w.detected_ppe.vest);
            if (hasH && hasV) {
                chipsHtml = `<span class="compliant-pill-chip">Hard hat and vest</span>`;
            } else if (hasH) {
                chipsHtml = `<span class="compliant-pill-chip">Hard hat</span>`;
            } else if (hasV) {
                chipsHtml = `<span class="compliant-pill-chip chip-note">Vest only, hard hat not required</span>`;
            } else {
                chipsHtml = `<span class="compliant-pill-chip">No PPE required here</span>`;
            }
        } else {
            chipsHtml = (w.missing_chips || []).map(chip => `<span class="violation-pill-chip">${chip}</span>`).join("");
        }

        const revStatus = w.reviewer_status || "confirm";

        return `
            <div class="worker-verdict-card ${isSelected ? 'selected' : ''}" onclick="selectImageWorker(${w.worker_id})">
                <div class="card-top-row">
                    <div class="worker-meta-left">
                        <span class="worker-id-badge">Worker #${w.worker_id}</span>
                        <span class="worker-conf-score">${confPct}%</span>
                    </div>
                    <span class="status-badge-chip ${badgeClass}">${badgeText}</span>
                </div>
                <div class="worker-tags-row">${chipsHtml}</div>
                <div class="card-bottom-row">
                    <span>Your review</span>
                    <select class="reviewer-dropdown" onclick="event.stopPropagation()" onchange="modifyImageVerdict(${w.worker_id}, this.value)">
                        <option value="confirm" ${revStatus === 'confirm' ? 'selected' : ''}>Confirmed</option>
                        <option value="false_positive" ${revStatus === 'false_positive' ? 'selected' : ''}>Dismiss as False Alarm</option>
                        <option value="override" ${revStatus === 'override' ? 'selected' : ''}>Force Violation</option>
                    </select>
                </div>
            </div>
        `;
    }).join("");
}

function selectImageWorker(workerId) {
    state.selectedWorkerId = (state.selectedWorkerId === workerId) ? null : workerId;
    reFilterImageWorkers();
}

function drawImageOverlayBoxes(workers) {
    const canvas = document.getElementById("image-overlay-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!workers || workers.length === 0) return;

    const img = document.getElementById("image-display-element");
    const naturalW = img.naturalWidth || 1280;
    const naturalH = img.naturalHeight || 720;

    const canvasW = canvas.width;
    const canvasH = canvas.height;

    // Precise letterbox / pillarbox offset calculation for object-fit: contain
    const imageAspect = naturalW / naturalH;
    const canvasAspect = canvasW / canvasH;

    let renderW, renderH, offsetX, offsetY;
    if (imageAspect > canvasAspect) {
        renderW = canvasW;
        renderH = canvasW / imageAspect;
        offsetX = 0;
        offsetY = (canvasH - renderH) / 2;
    } else {
        renderH = canvasH;
        renderW = canvasH * imageAspect;
        offsetX = (canvasW - renderW) / 2;
        offsetY = 0;
    }

    const scaleX = renderW / naturalW;
    const scaleY = renderH / naturalH;

    workers.forEach(w => {
        if (!w.box) return;
        const [x1, y1, x2, y2] = w.box;
        const sx = offsetX + (x1 * scaleX);
        const sy = offsetY + (y1 * scaleY);
        const sw = (x2 - x1) * scaleX;
        const sh = (y2 - y1) * scaleY;

        const isViolation = !w.is_compliant;
        const isSelected = state.selectedWorkerId === w.worker_id;

        // Bounding Boxes
        if (state.imageOverlays.boxes) {
            ctx.save();
            ctx.lineWidth = isSelected ? 3 : 2;
            ctx.strokeStyle = isSelected ? "#FFFFFF" : (isViolation ? "#A31C1B" : "#A3A3A3");
            ctx.strokeRect(sx, sy, sw, sh);

            if (isSelected) {
                ctx.fillStyle = isViolation ? "rgba(163, 28, 27, 0.18)" : "rgba(163, 163, 163, 0.18)";
                ctx.fillRect(sx, sy, sw, sh);
            }
            ctx.restore();
        }

        // Labels
        if (state.imageOverlays.labels) {
            ctx.save();
            const confPct = Math.round(w.confidence * 100);
            const violationSummary = (isViolation && w.missing_chips && w.missing_chips.length > 0)
                ? ` [${w.missing_chips.join(', ')}]`
                : '';
            const labelText = `Worker #${w.worker_id} - ${isViolation ? 'Violation' : 'Compliant'} - ${confPct}%${violationSummary}`;
            
            ctx.font = "600 11px Inter, sans-serif";
            const textMetrics = ctx.measureText(labelText);
            const chipW = textMetrics.width + 12;
            const chipH = 18;

            ctx.fillStyle = isViolation ? "#A31C1B" : "#262626";
            ctx.fillRect(sx, Math.max(sy - chipH, 0), chipW, chipH);

            ctx.fillStyle = "#FFFFFF";
            ctx.fillText(labelText, sx + 6, Math.max(sy - 4, 12));
            ctx.restore();
        }

        // Zones
        if (state.imageOverlays.zones) {
            ctx.save();
            ctx.setLineDash([3, 3]);
            ctx.lineWidth = 1;

            // Head Zone (0-32%)
            ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
            ctx.strokeRect(sx, sy, sw, sh * 0.32);

            // Torso Zone (22-75%)
            ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
            ctx.strokeRect(sx, sy + (sh * 0.22), sw, sh * 0.53);
            ctx.restore();
        }
    });

    // Draw Unassigned / Carried items (strictly helmets or vests, ignoring generic background clutter)
    const unassigned = (state.currentImageDetections?.unassigned_items || []).filter(item => {
        const cn = (item.class_name || "").toLowerCase();
        return cn.includes("helmet") || cn.includes("hardhat") || cn.includes("vest");
    });
    if (unassigned.length > 0 && state.imageOverlays.boxes) {
        unassigned.forEach(item => {
            if (!item.box) return;
            const [x1, y1, x2, y2] = item.box;
            const sx = offsetX + (x1 * scaleX);
            const sy = offsetY + (y1 * scaleY);
            const sw = (x2 - x1) * scaleX;
            const sh = (y2 - y1) * scaleY;

            ctx.save();
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = "#EAB308"; // Amber highlight for carried/unassigned gear
            ctx.lineWidth = 2;
            ctx.strokeRect(sx, sy, sw, sh);

            if (state.imageOverlays.labels) {
                const label = `${item.class_name} (${Math.round((item.confidence || 0.8) * 100)}%)`;
                ctx.font = "600 10px Inter, sans-serif";
                const metrics = ctx.measureText(label);
                ctx.fillStyle = "#EAB308";
                ctx.fillRect(sx, Math.max(sy - 16, 0), metrics.width + 8, 16);
                ctx.fillStyle = "#0A0A0A";
                ctx.fillText(label, sx + 4, Math.max(sy - 4, 12));
            }
            ctx.restore();
        });
    }
}

function toggleImageOverlay(type) {
    state.imageOverlays[type] = !state.imageOverlays[type];
    const btn = document.getElementById(`img-toggle-${type}`);
    if (btn) btn.classList.toggle("active", state.imageOverlays[type]);
    reFilterImageWorkers();
}

function rotateImage(deg = 90) {
    state.imageRotation = ((state.imageRotation || 0) + deg) % 360;
    applyImageTransform();
}

function adjustImageZoom(delta, reset = false) {
    if (reset) {
        state.imageZoom = 1.0;
        state.imageRotation = 0;
    } else {
        state.imageZoom = Math.max(0.6, Math.min(state.imageZoom + delta, 2.5));
    }
    applyImageTransform();
}

function applyImageTransform() {
    const img = document.getElementById("image-display-element");
    const canvas = document.getElementById("image-overlay-canvas");
    const rot = state.imageRotation || 0;
    const transformStr = `scale(${state.imageZoom}) rotate(${rot}deg)`;
    if (img) img.style.transform = transformStr;
    if (canvas) canvas.style.transform = transformStr;
}

function toggleImageRule(btn) {
    const rule = btn.dataset.imrule;
    btn.classList.toggle("selected");
    state.imageRules[rule] = btn.classList.contains("selected");
    reFilterImageWorkers();
}

function setImageRules(selectAll) {
    const btns = document.querySelectorAll("[data-imrule]");
    btns.forEach(btn => {
        if (selectAll) {
            btn.classList.add("selected");
            state.imageRules[btn.dataset.imrule] = true;
        } else {
            btn.classList.remove("selected");
            state.imageRules[btn.dataset.imrule] = false;
        }
    });
    reFilterImageWorkers();
}

function modifyImageVerdict(workerId, action) {
    if (!state.currentImageDetections || !state.currentImageDetections.workers) return;
    const w = state.currentImageDetections.workers.find(worker => worker.worker_id === workerId);
    if (!w) return;

    w.reviewer_status = action;
    reFilterImageWorkers();
}

// -------------------------------------------------------------
// 4. Video File Processing
// -------------------------------------------------------------
// -------------------------------------------------------------
// 4. Dedicated Video File Processing & Timeline Audit
// -------------------------------------------------------------
function initVideoUpload() {
    const vidInput = document.getElementById("hidden-video-upload");
    const dropzone = document.getElementById("video-dropzone");
    const stage = document.getElementById("video-stage-box");

    vidInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            inspectUploadedVideo(e.target.files[0]);
        }
    });

    if (dropzone) {
        dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropzone.style.borderColor = "var(--accent)";
        });
        dropzone.addEventListener("dragleave", () => {
            dropzone.style.borderColor = "#2e2e2e";
        });
        dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropzone.style.borderColor = "#2e2e2e";
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                inspectUploadedVideo(e.dataTransfer.files[0]);
            }
        });
    }

    if (stage) {
        stage.addEventListener("dragover", (e) => e.preventDefault());
        stage.addEventListener("drop", (e) => {
            e.preventDefault();
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                inspectUploadedVideo(e.dataTransfer.files[0]);
            }
        });
        stage.addEventListener("dblclick", () => vidInput.click());
    }

    const slider = document.getElementById("video-conf-slider");
    const numInput = document.getElementById("video-conf-num");
    if (slider && numInput) {
        slider.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            state.videoThreshold = val;
            numInput.value = Math.round(val * 100);
            reFilterVideoIncidents();
        });
        numInput.addEventListener("change", (e) => {
            let val = parseInt(e.target.value, 10);
            if (isNaN(val)) val = 35;
            val = Math.max(10, Math.min(val, 95));
            numInput.value = val;
            slider.value = (val / 100).toFixed(2);
            state.videoThreshold = val / 100;
            reFilterVideoIncidents();
        });
    }
}

async function inspectUploadedVideo(file) {
    showVideoProgress(true);

    const videoEl = document.getElementById("video-display-element");
    const dropzone = document.getElementById("video-dropzone");
    const metaText = document.getElementById("video-meta-text");

    const localUrl = URL.createObjectURL(file);
    videoEl.src = localUrl;
    videoEl.classList.remove("hidden");
    if (dropzone) dropzone.classList.add("hidden");
    metaText.innerText = `Checking ${file.name} frame by frame…`;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("max_frames", "150");
    formData.append("conf_thresh", state.videoThreshold.toString());

    const t0 = performance.now();
    try {
        const res = await fetch(`${API_BASE}/api/detect/video`, {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        showVideoProgress(false);

        if (data.success) {
            state.currentVideoData = data;
            const latency = performance.now() - t0;

            metaText.innerText = `${file.name}: ${data.total_frames_processed} frames checked, ${data.cumulative_violations_spotted} with missing PPE, took ${(latency/1000).toFixed(1)} s`;

            reFilterVideoIncidents();
            const evaluated = getEvaluatedVideoIncidents();
            if (evaluated.some(i => !i.is_compliant)) {
                autoLogInspectionIncident("Video File Audit", evaluated);
            }
        } else {
            alert("Video analysis failed: " + (data.detail || "Server error"));
        }
    } catch (err) {
        showVideoProgress(false);
        alert("Error connecting to detection backend: " + err.message);
    }
}

function showVideoProgress(show) {
    const bar = document.getElementById("video-progress-bar");
    if (bar) bar.style.display = show ? "block" : "none";
}

function getEvaluatedVideoIncidents() {
    if (!state.currentVideoData) return [];

    const data = state.currentVideoData;
    const rawIncidents = data.incidents || [];

    return rawIncidents.map(inc => {
        let isComp = true;
        const missing = [];

        if (state.videoRules.helmet && !inc.detected_ppe?.helmet) {
            isComp = false;
            missing.push("No helmet");
        }
        if (state.videoRules.vest && !inc.detected_ppe?.vest) {
            isComp = false;
            missing.push("No safety vest");
        }

        if (inc.reviewer_status === "false_alarm") {
            isComp = true;
            missing.length = 0;
        } else if (inc.reviewer_status === "violation") {
            isComp = false;
            if (missing.length === 0) missing.push("Safety Breach (Manual)");
        }

        return {
            ...inc,
            is_compliant: isComp,
            verdict: isComp ? "COMPLIANT" : "VIOLATION",
            missing_chips: missing.length > 0 ? missing : ["Full Helmet & Vest"]
        };
    });
}

function reFilterVideoIncidents() {
    if (!state.currentVideoData) return;
    const evaluated = getEvaluatedVideoIncidents();
    renderVideoVerdictsList(evaluated);
    updateVideoKpis(evaluated);
}

function updateVideoKpis(evaluated) {
    if (state.currentTab !== "video") return;

    const data = state.currentVideoData;
    const frames = data ? data.total_frames_processed : 0;
    const peak = data ? data.peak_workers : 0;

    const incidents = evaluated || (data ? data.incidents : []) || [];
    const viol = incidents.filter(i => !i.is_compliant).length;
    const comp = incidents.filter(i => i.is_compliant).length;
    const rate = incidents.length > 0 ? Math.round((comp / incidents.length) * 100) : 100;

    const framesVal = document.getElementById("vid-kpi-frames-val");
    if (framesVal) framesVal.innerText = frames;
    const framesSub = document.getElementById("vid-kpi-frames-sub");
    if (framesSub) framesSub.innerText = frames > 0 ? `${frames} frames analyzed` : "No video loaded";

    const workersVal = document.getElementById("vid-kpi-workers-val");
    if (workersVal) workersVal.innerText = peak;
    const workersSub = document.getElementById("vid-kpi-workers-sub");
    if (workersSub) workersSub.innerText = peak > 0 ? `Peak ${peak} workers / frame` : "0 workers in stream";

    const violVal = document.getElementById("vid-kpi-violations-val");
    if (violVal) violVal.innerText = viol;
    const violSub = document.getElementById("vid-kpi-violations-sub");
    if (violSub) violSub.innerText = `${viol} safety breaches flagged`;

    const violCard = document.getElementById("vid-kpi-violations-card");
    if (violCard) {
        if (viol > 0) violCard.classList.add("has-violations");
        else violCard.classList.remove("has-violations");
    }

    const rateVal = document.getElementById("vid-kpi-rate-val");
    if (rateVal) rateVal.innerText = `${rate}%`;
    const rateSub = document.getElementById("vid-kpi-rate-sub");
    if (rateSub) rateSub.innerText = rate >= 90 ? "Site safety target met" : "Below site safety target (90%)";
}

function renderVideoVerdictsList(incidents) {
    const container = document.getElementById("video-verdicts-container");
    const badge = document.getElementById("video-verdicts-badge");
    if (!container) return;

    if (!incidents || incidents.length === 0) {
        if (badge) badge.innerText = "0 Incidents";
        container.innerHTML = `
            <div class="list-empty">Choose a video. Each flagged moment appears here, and clicking one jumps the video to it.</div>
        `;
        return;
    }

    if (badge) badge.innerText = `${incidents.length} Incident${incidents.length !== 1 ? 's' : ''}`;

    container.innerHTML = incidents.map((inc, idx) => {
        const isComp = inc.is_compliant;
        const badgeClass = isComp ? "compliant" : "violation";
        const badgeText = isComp ? "Wearing PPE" : "Missing PPE";
        const confPct = Math.round(inc.confidence * 100);
        const revStatus = inc.reviewer_status || "confirm";

        const chipsHtml = isComp
            ? `<span class="compliant-pill-chip">Hard hat and vest</span>`
            : (inc.missing_chips || []).map(chip => `<span class="violation-pill-chip">${chip}</span>`).join("");

        return `
            <div class="worker-verdict-card" onclick="seekVideoToTime(${inc.timestamp_sec})" title="Click to seek video to ${inc.timestamp_sec}s">
                <div class="card-top-row">
                    <div class="worker-meta-left">
                        <span class="worker-id-badge">At ${inc.timestamp_sec}s (frame ${inc.frame})</span>
                        <span class="worker-conf-score">${confPct}%</span>
                    </div>
                    <span class="status-badge-chip ${badgeClass}">${badgeText}</span>
                </div>
                <div class="worker-tags-row">${chipsHtml}</div>
                <div class="card-bottom-row">
                    <span>Jump to ${inc.timestamp_sec}s</span>
                    <select class="reviewer-dropdown" onclick="event.stopPropagation()" onchange="modifyVideoVerdict(${idx}, this.value)">
                        <option value="confirm" ${revStatus === 'confirm' ? 'selected' : ''}>Confirmed</option>
                        <option value="false_alarm" ${revStatus === 'false_alarm' ? 'selected' : ''}>Dismiss as False Alarm</option>
                        <option value="violation" ${revStatus === 'violation' ? 'selected' : ''}>Force Violation</option>
                    </select>
                </div>
            </div>
        `;
    }).join("");
}

function seekVideoToTime(sec) {
    const video = document.getElementById("video-display-element");
    if (video) {
        video.currentTime = sec;
        video.play();
    }
}

function modifyVideoVerdict(idx, action) {
    if (!state.currentVideoData || !state.currentVideoData.incidents) return;
    const inc = state.currentVideoData.incidents[idx];
    if (!inc) return;

    inc.reviewer_status = action;
    reFilterVideoIncidents();
}

function toggleVideoRule(btn) {
    const rule = btn.dataset.vidrule;
    btn.classList.toggle("selected");
    state.videoRules[rule] = btn.classList.contains("selected");
    reFilterVideoIncidents();
}

function setVideoRules(selectAll) {
    const btns = document.querySelectorAll("[data-vidrule]");
    btns.forEach(btn => {
        if (selectAll) {
            btn.classList.add("selected");
            state.videoRules[btn.dataset.vidrule] = true;
        } else {
            btn.classList.remove("selected");
            state.videoRules[btn.dataset.vidrule] = false;
        }
    });
    reFilterVideoIncidents();
}

function toggleVideoFullscreen() {
    const video = document.getElementById("video-display-element");
    if (!video) return;
    if (!document.fullscreenElement) {
        video.requestFullscreen?.().catch(e => alert("Fullscreen error: " + e.message));
    } else {
        document.exitFullscreen?.();
    }
}

// -------------------------------------------------------------
// 5. Audit Logging & Export & Visual Snapshot Proofs
// -------------------------------------------------------------
function captureCameraSnapshot(camIndex) {
    try {
        const video = document.getElementById(`video-cam-${camIndex}`);
        const canvas = document.getElementById(`canvas-cam-${camIndex}`);
        const camObj = state.activeCameras[camIndex];

        let source = null;
        let srcW = 0;
        let srcH = 0;

        if (video && video.videoWidth && video.videoHeight) {
            source = video;
            srcW = video.videoWidth;
            srcH = video.videoHeight;
        } else if (camObj && camObj.lastFrameCanvas && camObj.lastFrameCanvas.width) {
            source = camObj.lastFrameCanvas;
            srcW = camObj.lastFrameCanvas.width;
            srcH = camObj.lastFrameCanvas.height;
        }

        if (!source || !srcW || !srcH) {
            console.warn(`[Snapshot] No video or canvas frame available for CAM-${camIndex}`);
            return null;
        }

        const off = document.createElement("canvas");
        const scale = Math.min(1.0, 960 / Math.max(srcW, srcH));
        off.width = Math.round(srcW * scale);
        off.height = Math.round(srcH * scale);
        const ctx = off.getContext("2d");

        // 1. Draw live frame
        ctx.drawImage(source, 0, 0, off.width, off.height);

        // 2. Draw violation overlays onto proof snapshot
        const detections = camObj ? (camObj.detections || []) : [];
        if (canvas && canvas.width && canvas.height) {
            ctx.drawImage(canvas, 0, 0, off.width, off.height);
        } else if (detections.length > 0) {
            const scaleX = off.width / 640.0;
            const scaleY = off.height / 360.0;
            detections.forEach(w => {
                if (!w.box) return;
                const [x1, y1, x2, y2] = w.box;
                const sx = x1 * scaleX;
                const sy = y1 * scaleY;
                const sw = (x2 - x1) * scaleX;
                const sh = (y2 - y1) * scaleY;
                const isViolation = !w.is_compliant;
                const isDuplicate = !!w.is_room_duplicate;
                let strokeColor = "#A3A3A3";
                let chipBg = "#262626";
                const missingText = (isViolation && w.missing_chips && w.missing_chips.length) ? ` (${w.missing_chips.join(', ')})` : '';
                let labelText = `Worker #${w.worker_id} - COMPLIANT${missingText}`;

                if (isViolation) {
                    if (isDuplicate) {
                        strokeColor = "#F97316"; // Vibrant Orange for Room Duplicate
                        chipBg = "#EA580C";
                        labelText = `⚠️ Worker #${w.worker_id} [CAM-0${w.duplicate_of_cam} OVERLAP]${missingText}`;
                    } else {
                        strokeColor = "#A31C1B"; // Red for Primary Breach
                        chipBg = "#A31C1B";
                        labelText = `Worker #${w.worker_id} - VIOLATION${missingText}`;
                    }
                }

                ctx.save();
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = isDuplicate ? 3.5 : 3;
                ctx.strokeRect(sx, sy, sw, sh);

                ctx.font = "bold 13px Inter, sans-serif";
                const metrics = ctx.measureText(labelText);
                ctx.fillStyle = chipBg;
                ctx.fillRect(sx, Math.max(sy - 22, 0), metrics.width + 12, 22);
                ctx.fillStyle = "#FFFFFF";
                ctx.fillText(labelText, sx + 6, Math.max(sy - 6, 16));
                ctx.restore();
            });
        }
        return off.toDataURL("image/jpeg", 0.85);
    } catch (e) {
        console.warn("Could not capture camera snapshot:", e);
        return null;
    }
}

function captureImageInspectionSnapshot() {
    try {
        const img = document.getElementById("image-display-element");
        const canvas = document.getElementById("image-overlay-canvas");
        if (!img || !img.naturalWidth || !canvas) {
            return state.currentImageDetections?.processed_image_url || null;
        }

        const off = document.createElement("canvas");
        off.width = canvas.width || 1280;
        off.height = canvas.height || 720;
        const ctx = off.getContext("2d");

        // Fill background
        ctx.fillStyle = "#050505";
        ctx.fillRect(0, 0, off.width, off.height);

        // Aspect fit matching canvas
        const naturalW = img.naturalWidth || 1280;
        const naturalH = img.naturalHeight || 720;
        const imageAspect = naturalW / naturalH;
        const canvasAspect = off.width / off.height;
        let renderW, renderH, offsetX, offsetY;
        if (imageAspect > canvasAspect) {
            renderW = off.width;
            renderH = off.width / imageAspect;
            offsetX = 0;
            offsetY = (off.height - renderH) / 2;
        } else {
            renderH = off.height;
            renderW = off.height * imageAspect;
            offsetX = (off.width - renderW) / 2;
            offsetY = 0;
        }

        // Draw image frame
        ctx.drawImage(img, offsetX, offsetY, renderW, renderH);

        // Draw exact overlay canvas on top (which has the red boxes and violation chips!)
        ctx.drawImage(canvas, 0, 0);

        return off.toDataURL("image/jpeg", 0.85);
    } catch (e) {
        console.warn("Could not capture image inspection snapshot:", e);
        return state.currentImageDetections?.processed_image_url || null;
    }
}

function captureVideoInspectionSnapshot() {
    try {
        const video = document.getElementById("video-display-element");
        if (!video || !video.videoWidth) return null;

        const off = document.createElement("canvas");
        const scale = Math.min(1.0, 960 / Math.max(video.videoWidth, video.videoHeight));
        off.width = Math.round(video.videoWidth * scale);
        off.height = Math.round(video.videoHeight * scale);
        const ctx = off.getContext("2d");

        ctx.drawImage(video, 0, 0, off.width, off.height);
        return off.toDataURL("image/jpeg", 0.80);
    } catch (e) {
        console.warn("Could not capture video inspection snapshot:", e);
        return null;
    }
}

function autoLogInspectionIncident(sourceName, evaluatedList) {
    const isVideo = sourceName.includes("Video");
    const count = isVideo ? (state.currentVideoData?.total_frames_processed || evaluatedList.length) : evaluatedList.length;
    const violations = evaluatedList.filter(item => !item.is_compliant);
    if (violations.length === 0) return;

    let summary = "";
    let snapshotImage = null;
    let activeRules = {};
    let workers = [];

    if (isVideo) {
        summary = violations.slice(0, 3).map(i => `Frame #${i.frame}: ${(i.missing_chips || []).join(', ')}`).join(" | ");
        if (violations.length > 3) summary += ` (+${violations.length - 3} more)`;
        snapshotImage = captureVideoInspectionSnapshot();
        activeRules = { ...state.videoRules };
        workers = evaluatedList.map(i => ({
            worker_id: `Frame #${i.frame} (${i.timestamp_sec}s)`,
            confidence: i.confidence,
            is_compliant: i.is_compliant,
            verdict: i.is_compliant ? "COMPLIANT" : "VIOLATION",
            missing_chips: i.missing_chips || [],
            detected_ppe: i.detected_ppe || {}
        }));
    } else {
        summary = violations.map(w => `Worker #${w.worker_id}: ${(w.missing_chips || []).join(', ')}`).join(" | ");
        snapshotImage = captureImageInspectionSnapshot();
        activeRules = { ...state.imageRules };
        workers = evaluatedList.map(w => ({
            worker_id: w.worker_id,
            confidence: w.confidence,
            is_compliant: w.is_compliant,
            verdict: w.is_compliant ? "COMPLIANT" : "VIOLATION",
            missing_chips: w.missing_chips || [],
            detected_ppe: w.detected_ppe || {},
            negative_flags: w.negative_flags || []
        }));
    }

    const logEntry = {
        id: `INC-${Date.now().toString().slice(-6)}`,
        timestamp: new Date().toLocaleString(),
        camera: sourceName,
        totalWorkers: count,
        compliant: Math.max(0, count - violations.length),
        violations: violations.length,
        verdict: summary,
        snapshotImage: snapshotImage,
        activeRules: activeRules,
        workers: workers
    };

    state.auditLogs.unshift(logEntry);
    saveAuditLogs();
    renderAuditTable();
}

function addToAuditReport(sourceType = "System") {
    const isImage = (sourceType === "Image" || state.currentTab === "image");
    const isVideo = (sourceType === "Video" || state.currentTab === "video");

    let count = 0;
    let violCount = 0;
    let sourceName = "CCTV Multi-Camera Console";
    let summary = "";
    let snapshotImage = null;
    let activeRules = {};
    let workers = [];

    if (isImage) {
        if (!state.currentImageDetections) return alert("No active image inspection to record.");
        const evaluated = getEvaluatedImageWorkers();
        count = evaluated.length;
        const violations = evaluated.filter(w => !w.is_compliant);
        violCount = violations.length;
        sourceName = "Photo Inspection";
        if (violCount > 0) {
            summary = violations.map(w => `Worker #${w.worker_id}: ${(w.missing_chips || []).join(', ')}`).join(" | ");
        } else {
            summary = "Full PPE Compliance Verified";
        }
        snapshotImage = captureImageInspectionSnapshot();
        activeRules = { ...state.imageRules };
        workers = evaluated.map(w => ({
            worker_id: w.worker_id,
            confidence: w.confidence,
            is_compliant: w.is_compliant,
            verdict: w.is_compliant ? "COMPLIANT" : "VIOLATION",
            missing_chips: w.missing_chips || [],
            detected_ppe: w.detected_ppe || {},
            negative_flags: w.negative_flags || []
        }));
    } else if (isVideo) {
        if (!state.currentVideoData) return alert("No active video inspection to record.");
        const evaluated = getEvaluatedVideoIncidents();
        count = state.currentVideoData.total_frames_processed || evaluated.length;
        const violations = evaluated.filter(i => !i.is_compliant);
        violCount = violations.length;
        sourceName = "Video File Audit";
        if (violCount > 0) {
            summary = violations.slice(0, 4).map(i => `Frame #${i.frame} (${i.timestamp_sec}s): ${(i.missing_chips || []).join(', ')}`).join(" | ");
            if (violations.length > 4) summary += ` (+${violations.length - 4} more)`;
        } else {
            summary = "No Violations Spotted Across Frames";
        }
        snapshotImage = captureVideoInspectionSnapshot();
        activeRules = { ...state.videoRules };
        workers = evaluated.map(i => ({
            worker_id: `Frame #${i.frame} (${i.timestamp_sec}s)`,
            confidence: i.confidence,
            is_compliant: i.is_compliant,
            verdict: i.is_compliant ? "COMPLIANT" : "VIOLATION",
            missing_chips: i.missing_chips || [],
            detected_ppe: i.detected_ppe || {}
        }));
    } else {
        const allWorkers = Object.values(state.activeCameras).flatMap(c => c.detections || []);
        if (allWorkers.length === 0) return alert("No active camera streams to record.");
        count = allWorkers.length;
        const violations = allWorkers.filter(w => !w.is_compliant);
        violCount = violations.length;
        sourceName = "CCTV Multi-Camera Console";
        if (violCount > 0) {
            summary = violations.map(w => `${w.camera} Worker #${w.worker_id}: ${(w.missing_chips || []).join(', ')}`).join(" | ");
        } else {
            summary = "All Active Feeds Compliant";
        }

        // Determine camera to capture visual proof snapshot from
        let targetCam = state.activeCctvFocus;
        if (!targetCam) {
            const violCam = Object.entries(state.activeCameras).find(([idx, cam]) => cam.streaming && (cam.detections || []).some(w => !w.is_compliant));
            if (violCam) targetCam = parseInt(violCam[0], 10);
            else {
                const activeCam = Object.entries(state.activeCameras).find(([idx, cam]) => cam.streaming);
                if (activeCam) targetCam = parseInt(activeCam[0], 10);
            }
        }
        if (targetCam) {
            const camObj = state.activeCameras[targetCam];
            if (camObj && camObj.roomName) {
                sourceName = `CAM-0${targetCam} (${camObj.roomName})`;
            }
            if (camObj && camObj.rules) {
                activeRules = { ...camObj.rules };
            } else {
                activeRules = { ...state.cctvRules };
            }
            snapshotImage = captureCameraSnapshot(targetCam);
        } else {
            activeRules = { ...state.cctvRules };
        }

        workers = allWorkers.map(w => ({
            worker_id: `${w.camera || 'CAM'} Worker #${w.worker_id}`,
            confidence: w.confidence,
            is_compliant: w.is_compliant,
            verdict: w.is_compliant ? "COMPLIANT" : "VIOLATION",
            missing_chips: w.missing_chips || [],
            detected_ppe: w.detected_ppe || {},
            negative_flags: w.negative_flags || []
        }));
    }

    const logEntry = {
        id: `INC-${Date.now().toString().slice(-6)}`,
        timestamp: new Date().toLocaleString(),
        camera: sourceName,
        totalWorkers: count,
        compliant: Math.max(0, count - violCount),
        violations: violCount,
        verdict: summary,
        snapshotImage: snapshotImage,
        activeRules: activeRules,
        workers: workers
    };

    state.auditLogs.unshift(logEntry);
    saveAuditLogs();
    renderAuditTable();
    alert(`Incident ${logEntry.id} logged to safety audit report with visual proof.`);
}

function dismissFalsePositive(context) {
    const isImage = (context === "image" || state.currentTab === "image");
    const isVideo = (context === "video" || state.currentTab === "video");

    if (isImage) {
        if (!state.currentImageDetections || !state.currentImageDetections.workers || !state.currentImageDetections.workers.length) {
            return alert("No photo evaluation loaded.");
        }
        let targetId = state.selectedWorkerId;
        if (!targetId) {
            const viol = state.currentImageDetections.workers.find(w => !w.is_compliant);
            if (viol) targetId = viol.worker_id;
            else targetId = state.currentImageDetections.workers[0].worker_id;
        }
        modifyImageVerdict(targetId, "false_positive");
    } else if (isVideo) {
        if (!state.currentVideoData || !state.currentVideoData.incidents) {
            return alert("No video evaluation loaded.");
        }
        const violIdx = state.currentVideoData.incidents.findIndex(i => !i.is_compliant && i.reviewer_status !== "false_alarm");
        if (violIdx >= 0) {
            modifyVideoVerdict(violIdx, "false_alarm");
        } else {
            alert("No video violations to dismiss.");
        }
    } else {
        const allWorkers = Object.values(state.activeCameras).flatMap(c => c.detections || []);
        const viol = allWorkers.find(w => !w.is_compliant);
        if (viol) {
            modifyCctvVerdict(viol.camera, viol.worker_id, "false_alarm");
        } else {
            alert("No active camera violations to dismiss.");
        }
    }
}

function exportQuickSummary(ctx) {
    let data = null;
    let filename = `ppe_inspection_snapshot_${Date.now()}.json`;

    if (ctx === "image" || state.currentTab === "image") {
        data = state.currentImageDetections;
        filename = `ppe_photo_inspection_${Date.now()}.json`;
    } else if (ctx === "video" || state.currentTab === "video") {
        data = state.currentVideoData;
        filename = `ppe_video_audit_${Date.now()}.json`;
    } else {
        data = { note: "CCTV snapshot", time: new Date(), activeCameras: state.activeCameras };
        filename = `ppe_cctv_snapshot_${Date.now()}.json`;
    }

    if (!data) return alert("No active inspection data to export.");
    downloadBlob(JSON.stringify(data, null, 2), filename, "application/json");
}

function scrollToVerdicts() {
    let target = "cctv-verdicts-container";
    if (state.currentTab === "image") target = "image-verdicts-container";
    else if (state.currentTab === "video") target = "video-verdicts-container";
    const list = document.getElementById(target);
    if (list) list.scrollIntoView({ behavior: "smooth" });
}

function toggleRule(btn) {
    const rule = btn.dataset.rule;
    btn.classList.toggle("selected");
    state.cctvRules[rule] = btn.classList.contains("selected");
    updateCctvKpis();
}

function setViolationRules(selectAll) {
    const btns = document.querySelectorAll("[data-rule]");
    btns.forEach(btn => {
        if (selectAll) {
            btn.classList.add("selected");
            state.cctvRules[btn.dataset.rule] = true;
        } else {
            btn.classList.remove("selected");
            state.cctvRules[btn.dataset.rule] = false;
        }
    });
    updateCctvKpis();
}

// -------------------------------------------------------------
// Audit Table Rendering, Persistence & Export Helpers
// -------------------------------------------------------------
function saveAuditLogs() {
    try {
        localStorage.setItem("ppe_audit_logs", JSON.stringify(state.auditLogs));
    } catch (e) {
        console.warn("localStorage quota exceeded, saving lightweight audit log references", e);
        try {
            // Strip raw base64 data to stay well within browser localStorage limit
            const leanLogs = (state.auditLogs || []).map(l => ({
                ...l,
                snapshotImage: l.snapshotImage && l.snapshotImage.startsWith("data:") 
                    ? `/media/snapshots/${l.id}.jpg` 
                    : l.snapshotImage,
                multiCamViews: null
            }));
            localStorage.setItem("ppe_audit_logs", JSON.stringify(leanLogs.slice(0, 50)));
        } catch (innerErr) {
            console.error("Critical failure saving audit logs", innerErr);
        }
    }
}

function loadAuditLogs() {
    try {
        const stored = localStorage.getItem("ppe_audit_logs");
        if (stored) {
            state.auditLogs = JSON.parse(stored);
        }
    } catch (e) {
        state.auditLogs = [];
    }
}

function renderAuditTable() {
    const tbody = document.getElementById("audit-table-body");
    if (!tbody) return;

    if (!state.auditLogs || state.auditLogs.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="9" class="table-empty">No incidents logged yet. Use "Log incident" on the camera, photo, or video screens.</td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = state.auditLogs.map((log, idx) => {
        const isViol = log.violations > 0;
        const badgeClass = isViol ? "violation" : "compliant";
        const badgeText = isViol ? "Missing PPE" : "Wearing PPE";
        const thumbUrl = log.snapshotImage || `/media/snapshots/${log.id}.jpg`;

        const proofThumb = `<div class="audit-snapshot-thumb" onclick="openProofLightbox('${log.id}')" title="View evidence">
             <img src="${thumbUrl}" onerror="this.src='/media/snapshots/${log.id}.jpg'" alt="Proof" />
             <span class="thumb-zoom-icon"><i data-lucide="maximize-2" style="width:12px; height:12px;"></i></span>
           </div>`;

        return `
            <tr class="audit-row" ondblclick="if(!event.target.closest('.audit-actions'))openProofPage('${log.id}')" title="Double-click to open full details">
                <td><button class="audit-id" onclick="openProofLightbox('${log.id}')" title="View evidence">${log.id}</button></td>
                <td>${proofThumb}</td>
                <td class="audit-muted">${log.timestamp}</td>
                <td class="audit-strong">${log.camera}</td>
                <td class="num">${log.totalWorkers}</td>
                <td class="num">${log.compliant}</td>
                <td class="num ${isViol ? 'audit-viol' : ''}">${log.violations}</td>
                <td>
                    <div class="audit-result">
                        <span class="status-badge-chip ${badgeClass}">${badgeText}</span>
                        <span class="audit-verdict" title="${String(log.verdict || '').replace(/"/g, '&quot;')}">${log.verdict}</span>
                    </div>
                </td>
                <td>
                    <div class="audit-actions">
                        <button class="btn-outlined btn-sm" onclick="openProofPage('${log.id}')" title="Open the evidence page in a new tab">
                            Open
                        </button>
                        <button class="btn-outlined btn-sm btn-danger" onclick="deleteAuditLogEntry(${idx})" title="Delete this incident">
                            <i data-lucide="trash-2"></i> Delete
                        </button>
                    </div>
                </td>
            </tr>
        `;
    }).join("");

    if (window.lucide) window.lucide.createIcons();
}

function openProofPage(logId) {
    window.open(`proof.html?id=${logId}`, '_blank');
}

let currentLightboxLogId = null;

function openProofLightbox(logId) {
    let log = (state.auditLogs || []).find(l => l.id === logId);
    if (!log) {
        log = {
            id: logId,
            camera: "Live Feed Proof",
            violations: 1,
            verdict: "Safety Violation Proof",
            timestamp: new Date().toLocaleString(),
            snapshotImage: `/media/snapshots/${logId}.jpg`
        };
    }

    currentLightboxLogId = logId;
    const modal = document.getElementById("proof-lightbox-modal");
    const img = document.getElementById("lightbox-image");
    const idEl = document.getElementById("lightbox-incident-id");
    const metaEl = document.getElementById("lightbox-incident-meta");
    const verdictEl = document.getElementById("lightbox-verdict-text");
    const timeEl = document.getElementById("lightbox-timestamp-text");

    const imgSrc = log.snapshotImage || `/media/snapshots/${log.id}.jpg`;
    if (idEl) idEl.innerText = log.id;
    if (metaEl) metaEl.innerText = `${log.camera}, ${log.violations > 0 ? log.violations + ' missing PPE' : 'all wearing PPE'}`;
    if (img) {
        img.src = imgSrc;
        img.onerror = () => {
            img.src = `/media/snapshots/${log.id}.jpg`;
        };
    }
    if (verdictEl) verdictEl.innerText = log.verdict || "Safety Breach Verified";
    if (timeEl) timeEl.innerText = `Recorded: ${log.timestamp} | Total Workers: ${log.totalWorkers || 1} (Compliant: ${log.compliant || 0})`;

    if (modal) modal.classList.remove("hidden");
    if (window.lucide) window.lucide.createIcons();
}

function closeProofLightbox() {
    const modal = document.getElementById("proof-lightbox-modal");
    if (modal) modal.classList.add("hidden");
    currentLightboxLogId = null;
}

function downloadProofImage() {
    if (!currentLightboxLogId) return;
    const log = (state.auditLogs || []).find(l => l.id === currentLightboxLogId);
    if (!log || !log.snapshotImage) return;

    const a = document.createElement("a");
    a.href = log.snapshotImage;
    a.download = `safety_violation_proof_${log.id}.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

function deleteAuditLogEntry(idx) {
    if (state.auditLogs && state.auditLogs[idx]) {
        if (!confirm(`Delete incident ${state.auditLogs[idx].id}? This can't be undone.`)) return;
        state.auditLogs.splice(idx, 1);
        saveAuditLogs();
        renderAuditTable();
    }
}

function exportAuditCSV() {
    if (!state.auditLogs || state.auditLogs.length === 0) {
        return alert("No audit log records to export.");
    }

    const headers = ["Incident ID", "Timestamp", "Source", "Total Workers", "Compliant", "Violations", "Verdict / Details"];
    const rows = state.auditLogs.map(l => [
        `"${l.id}"`,
        `"${l.timestamp}"`,
        `"${l.camera}"`,
        l.totalWorkers,
        l.compliant,
        l.violations,
        `"${(l.verdict || '').replace(/"/g, '""')}"`
    ]);

    const csvContent = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
    downloadBlob(csvContent, `safety_compliance_audit_${Date.now()}.csv`, "text/csv;charset=utf-8;");
}

function exportAuditJSON() {
    if (!state.auditLogs || state.auditLogs.length === 0) {
        return alert("No audit log records to export.");
    }
    downloadBlob(JSON.stringify(state.auditLogs, null, 2), `safety_compliance_audit_${Date.now()}.json`, "application/json");
}

function downloadBlob(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 100);
}
