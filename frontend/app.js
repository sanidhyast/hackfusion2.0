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
    activeCameras: {
        1: { streaming: false, stream: null, interval: null, label: "CAM-01", roomName: "Zone A - Main Scaffold", rules: { helmet: true, vest: true, helmet_arm: true, gloves: false, boots: false, goggles: false }, detections: [] },
        2: { streaming: false, stream: null, interval: null, label: "CAM-02", roomName: "Zone B - Material Unloading", rules: { helmet: true, vest: true, helmet_arm: true, gloves: false, boots: false, goggles: false }, detections: [] },
        3: { streaming: false, stream: null, interval: null, label: "CAM-03", roomName: "Zone C - Tower Crane Base", rules: { helmet: true, vest: true, helmet_arm: true, gloves: false, boots: false, goggles: false }, detections: [] },
        4: { streaming: false, stream: null, interval: null, label: "CAM-04", roomName: "Gate 2 - Personnel Entry", rules: { helmet: true, vest: true, helmet_arm: true, gloves: false, boots: false, goggles: false }, detections: [] }
    },
    cctvThreshold: 0.35,

    // Image Inspection State (Starts Clean)
    currentImageDetections: null,
    imageThreshold: 0.35,
    imageZoom: 1.0,
    imageRotation: 0,
    selectedWorkerId: null,

    // Image Overlays
    imageOverlays: {
        boxes: true,
        labels: true,
        zones: true
    },

    // Enforcement Rules for CCTV
    cctvRules: {
        helmet: true,
        vest: true,
        helmet_arm: true,
        gloves: false,
        boots: false,
        goggles: false,
        other: false
    },

    // Enforcement Rules for Image Inspection
    imageRules: {
        helmet: true,
        vest: true,
        helmet_arm: true,
        gloves: false,
        boots: false,
        goggles: false,
        other: false
    },

    // Video Audit State
    currentVideoData: null,
    videoThreshold: 0.35,
    videoRules: {
        helmet: true,
        vest: true,
        helmet_arm: true,
        gloves: false,
        boots: false,
        goggles: false,
        other: false
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
                    updateImageKpis();
                } else if (state.currentTab === "cctv") {
                    syncAllCanvases();
                    updateCctvKpis();
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
    const camRules = camObj.rules || { helmet: true, vest: true, helmet_arm: true, gloves: false, boots: false, goggles: false };
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

        meta.innerHTML = `<div class="live-dot" style="margin-right:2px; background:var(--accent);"></div> LIVE FEED`;

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
        statusFooter.innerText = "STANDBY";
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
    meta.innerHTML = `<div class="live-dot" style="margin-right:2px; background:var(--accent);"></div> DEMO FEED`;

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

function handleCamInferenceResult(camIndex, data) {
    const camObj = state.activeCameras[camIndex];
    const rawWorkers = data.workers || [];

    // Filter by active rules for this specific camera (falls back to global cctvRules)
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
        const hasCarriedHelmet = (w.negative_flags && w.negative_flags.includes("helmet_under_arm")) ||
                                 (w.missing_ppe && w.missing_ppe.includes("helmet_under_arm"));
        if (rules.helmet_arm && hasCarriedHelmet) {
            isCompliant = false;
            missing.push("Helmet Under Arm");
        }
        if (rules.gloves && !w.detected_ppe.gloves) {
            isCompliant = false;
            missing.push("No gloves");
        }
        if (rules.boots && !w.detected_ppe.boots) {
            isCompliant = false;
            missing.push("No boots");
        }
        if (rules.goggles && !w.detected_ppe.goggles) {
            isCompliant = false;
            missing.push("No eye protection");
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

    // Update tile footer
    const workersFooter = document.getElementById(`footer-cam-${camIndex}-workers`);
    const statusFooter = document.getElementById(`footer-cam-${camIndex}-status`);
    const tile = document.getElementById(`cam-tile-${camIndex}`);

    const violCount = evaluated.filter(w => !w.is_compliant).length;
    workersFooter.innerText = `Workers: ${evaluated.length}`;

    if (violCount > 0) {
        statusFooter.className = "status-tag-chip violation";
        statusFooter.innerText = `🚨 ${violCount} VIOLATION`;
        tile.classList.add("has-violation");
    } else {
        statusFooter.className = "status-tag-chip compliant";
        statusFooter.innerText = "COMPLIANT";
        tile.classList.remove("has-violation");
    }

    // Draw live bounding boxes on tile canvas
    drawCanvasBoxes(`canvas-cam-${camIndex}`, evaluated);

    // Aggregate KPIs across all active CCTV feeds
    updateCctvKpis();
}

function updateCctvKpis() {
    let total = 0;
    let compliant = 0;
    let violations = 0;
    const allWorkers = [];

    Object.keys(state.activeCameras).forEach(idx => {
        const cam = state.activeCameras[idx];
        if (cam.streaming && cam.detections) {
            total += cam.detections.length;
            const comp = cam.detections.filter(w => w.is_compliant).length;
            compliant += comp;
            violations += (cam.detections.length - comp);
            allWorkers.push(...cam.detections);
        }
    });

    const activeCount = Object.values(state.activeCameras).filter(c => c.streaming).length;
    const rate = total > 0 ? Math.round((compliant / total) * 100) : 100;

    document.getElementById("kpi-total-val").innerText = total;
    document.getElementById("kpi-total-sub").innerText = activeCount > 0 
        ? `${total} workers across ${activeCount} active camera${activeCount > 1 ? 's' : ''}`
        : "No active streams connected";

    document.getElementById("kpi-compliant-val").innerText = compliant;
    document.getElementById("kpi-compliant-sub").innerText = `${compliant} of ${total} workers compliant`;

    const violCard = document.getElementById("kpi-violations-card");
    document.getElementById("kpi-violations-val").innerText = violations;
    if (violations > 0) {
        violCard.classList.add("has-violations");
        document.getElementById("kpi-violations-sub").innerText = `${violations} active safety breaches`;
    } else {
        violCard.classList.remove("has-violations");
        document.getElementById("kpi-violations-sub").innerText = `0 active safety breaches`;
    }

    document.getElementById("kpi-rate-val").innerText = `${rate}%`;
    document.getElementById("kpi-rate-sub").innerText = rate >= 90 ? "Site safety target met" : "Below site safety target (90%)";

    // Update Right Panel CCTV verdicts list
    renderCctvVerdictsList(allWorkers);

    // Slim alert banner
    const banner = document.getElementById("violation-alert-banner");
    const bannerText = document.getElementById("alert-banner-text");
    if (violations > 0) {
        const firstViol = allWorkers.find(w => !w.is_compliant);
        bannerText.innerText = `${violations} violation detected - ${firstViol.camera} Worker #${firstViol.worker_id}: ${firstViol.missing_chips.join(', ')}`;
        banner.classList.remove("hidden");
    } else {
        banner.classList.add("hidden");
    }
}

function renderCctvVerdictsList(workers) {
    const container = document.getElementById("cctv-verdicts-container");
    const badge = document.getElementById("cctv-verdicts-badge");
    const activeCount = Object.values(state.activeCameras).filter(c => c.streaming).length;

    badge.innerText = `${activeCount} Active Feed${activeCount !== 1 ? 's' : ''}`;

    if (!workers || workers.length === 0) {
        container.innerHTML = `
            <div style="padding:32px 12px; text-align:center; color:var(--text-disabled); font-size:12px;">
                ${activeCount === 0 ? "No cameras connected. Click on any screen above to start camera." : "Camera streaming. No workers currently in view."}
            </div>
        `;
        return;
    }

    container.innerHTML = workers.map(w => {
        const isComp = w.is_compliant;
        const badgeClass = isComp ? "compliant" : "violation";
        const badgeText = isComp ? "COMPLIANT" : "VIOLATION";
        const conf = Math.round(w.confidence * 100);

        let chipsHtml = "";
        if (isComp) {
            const hasH = !!(w.detected_ppe && w.detected_ppe.helmet);
            const hasV = !!(w.detected_ppe && w.detected_ppe.vest);
            if (hasH && hasV) {
                chipsHtml = `<span class="compliant-pill-chip">Full Helmet & Vest</span>`;
            } else if (hasH) {
                chipsHtml = `<span class="compliant-pill-chip">Helmet Worn</span>`;
            } else if (hasV) {
                chipsHtml = `<span class="compliant-pill-chip" style="background:rgba(234,179,8,0.12); color:#FACC15; border-color:rgba(234,179,8,0.3);">Vest Worn (No Helmet)</span>`;
            } else {
                chipsHtml = `<span class="compliant-pill-chip" style="background:#262626; color:#A3A3A3; border-color:#404040;">No PPE (Rules Inactive)</span>`;
            }
        } else {
            chipsHtml = (w.missing_chips || []).map(c => `<span class="violation-pill-chip">${c}</span>`).join("");
        }

        return `
            <div class="worker-verdict-card">
                <div class="card-top-row">
                    <div class="worker-meta-left">
                        <span class="worker-id-badge">${w.camera} • Worker #${w.worker_id}</span>
                        <span class="worker-conf-score">${conf}%</span>
                    </div>
                    <span class="status-badge-chip ${badgeClass}">${badgeText}</span>
                </div>
                <div class="worker-tags-row">${chipsHtml}</div>
                <div class="card-bottom-row">
                    <span>${w.camera} • Live Stream</span>
                    <select class="reviewer-dropdown" onclick="event.stopPropagation()" onchange="modifyCctvVerdict('${w.camera}', ${w.worker_id}, this.value)">
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
    if (el) el.innerText = `${activeCount} of 4 Cameras Active`;
}

function updateLiveBadge() {
    const activeCount = Object.values(state.activeCameras).filter(c => c.streaming).length;
    const badge = document.getElementById("top-live-badge");
    const text = document.getElementById("top-live-text");
    if (activeCount > 0) {
        badge.classList.add("active");
        text.innerText = "LIVE";
    } else {
        badge.classList.remove("active");
        text.innerText = "STANDBY";
    }
}

function toggleQuadFocus(tileNum) {
    const quad = document.getElementById("cctv-quad-grid");
    const tiles = document.querySelectorAll(".cctv-tile");

    if (tileNum === null || state.activeCctvFocus === tileNum) {
        state.activeCctvFocus = null;
        quad.classList.remove("single-focus");
        tiles.forEach(t => t.classList.remove("active-focus"));
    } else {
        state.activeCctvFocus = tileNum;
        quad.classList.add("single-focus");
        tiles.forEach(t => t.classList.remove("active-focus"));
        const target = document.getElementById(`cam-tile-${tileNum}`);
        if (target) target.classList.add("active-focus");
    }
    setTimeout(syncAllCanvases, 100);
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
        }
    });
    syncImageCanvas();
}

function syncCanvasDimensions(canvasId, parentEl) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !parentEl) return;
    canvas.width = parentEl.clientWidth || 300;
    canvas.height = parentEl.clientHeight || 180;
}

function drawCanvasBoxes(canvasId, workers) {
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
        const strokeColor = isViolation ? "#A31C1B" : "#A3A3A3";

        // 1. Worker Outer Bounding Box
        ctx.save();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2;
        ctx.strokeRect(sx, sy, sw, sh);

        // 2. Anatomical Head Zone (0% - 32%) & Torso Zone (22% - 75%)
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
        ctx.strokeRect(sx, sy, sw, sh * 0.32); // Head zone

        ctx.strokeStyle = "rgba(255, 255, 255, 0.20)";
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
        const missingText = w.missing_chips && w.missing_chips.length > 0 
            ? ` (${w.missing_chips.join(', ')})` 
            : ' (Full PPE)';
        const labelText = `Worker #${w.worker_id} - ${isViolation ? 'VIOLATION' : 'OK'}${missingText}`;

        ctx.font = "600 11px Inter, sans-serif";
        const textMetrics = ctx.measureText(labelText);
        const chipW = textMetrics.width + 12;
        const chipH = 18;

        ctx.fillStyle = isViolation ? "#A31C1B" : "#262626";
        ctx.fillRect(sx, Math.max(sy - chipH, 0), chipW, chipH);

        ctx.fillStyle = "#FFFFFF";
        ctx.fillText(labelText, sx + 6, Math.max(sy - 4, 12));
        ctx.restore();
    });
}

// -------------------------------------------------------------
// 3. Dedicated Image Inspection (Starts Clean)
// -------------------------------------------------------------
function initImageControls() {
    const fileInput = document.getElementById("hidden-image-upload");
    const dropzone = document.getElementById("image-dropzone");

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            inspectUploadedImage(e.target.files[0]);
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
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            inspectUploadedImage(e.dataTransfer.files[0]);
        }
    });

    const stage = document.getElementById("image-stage-box");
    if (stage) {
        stage.addEventListener("dragover", (e) => {
            e.preventDefault();
        });
        stage.addEventListener("drop", (e) => {
            e.preventDefault();
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                inspectUploadedImage(e.dataTransfer.files[0]);
            }
        });
        stage.addEventListener("dblclick", () => {
            fileInput.click();
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
            imageConfDebounce = setTimeout(() => inspectUploadedImage(state.currentImageFile), 350);
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
            imageConfDebounce = setTimeout(() => inspectUploadedImage(state.currentImageFile), 350);
        }
    });
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
    state.currentImageFile = file;
    showImageProgress(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("return_base64", "true");
    formData.append("conf_thresh", state.imageThreshold.toString());

    const t0 = performance.now();
    try {
        const res = await fetch(`${API_BASE}/api/detect/image`, {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        state.latencyMs = performance.now() - t0;
        state.fps = 1000.0 / Math.max(state.latencyMs, 1.0);
        showImageProgress(false);

        if (data.success) {
            state.currentImageDetections = data;

            const img = document.getElementById("image-display-element");
            const dropzone = document.getElementById("image-dropzone");

            try {
                img.src = URL.createObjectURL(file);
            } catch (e) {
                img.src = data.annotated_image_base64;
            }
            img.classList.remove("hidden");
            dropzone.classList.add("hidden");

            document.getElementById("image-meta-text").innerText = `${file.name} • ${data.total_workers} Workers Evaluated • ${state.latencyMs.toFixed(1)} ms`;

            img.onload = () => {
                state.imageZoom = 1.0;
                state.imageRotation = 0;
                applyImageTransform();
                syncImageCanvas();
                reFilterImageWorkers();
                const evaluated = getEvaluatedImageWorkers();
                if (evaluated.some(w => !w.is_compliant)) {
                    autoLogInspectionIncident("Photo Inspection", evaluated);
                }
            };
        } else {
            alert("Inspection failed: " + (data.detail || "Server error"));
        }
    } catch (err) {
        showImageProgress(false);
        alert("Error connecting to detection backend: " + err.message);
    }
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

        if (state.imageRules.helmet && !w.detected_ppe.helmet) {
            isCompliant = false;
            missingChips.push("No helmet");
        }
        if (state.imageRules.vest && !w.detected_ppe.vest) {
            isCompliant = false;
            missingChips.push("No vest");
        }
        // Check for carried helmet (only evaluated if the worker does NOT already have a worn helmet on their head)
        let hasCarriedHelmet = false;
        if (!w.detected_ppe.helmet) {
            hasCarriedHelmet = (w.negative_flags && w.negative_flags.includes("helmet_under_arm")) ||
                               (w.missing_ppe && w.missing_ppe.includes("helmet_under_arm"));

            if (!hasCarriedHelmet && data.unassigned_items && data.unassigned_items.length && w.box) {
                const [wx1, wy1, wx2, wy2] = w.box;
                const ww = Math.max(1, wx2 - wx1);
                const wh = Math.max(1, wy2 - wy1);
                const czX1 = wx1 - 0.35 * ww;
                const czY1 = wy1 + 0.18 * wh;
                const czX2 = wx2 + 0.35 * ww;
                const czY2 = wy2 + 0.30 * wh;

                for (const item of data.unassigned_items) {
                    const cname = (item.class_name || "").toLowerCase();
                    const isCarriedCandidate = cname.includes("helmet") || cname.includes("hardhat");
                    if (isCarriedCandidate) {
                        const [ix1, iy1, ix2, iy2] = item.box;
                        const interX1 = Math.max(czX1, ix1);
                        const interY1 = Math.max(czY1, iy1);
                        const interX2 = Math.min(czX2, ix2);
                        const interY2 = Math.min(czY2, iy2);
                        const interW = Math.max(0, interX2 - interX1);
                        const interH = Math.max(0, interY2 - interY1);
                        const itemW = Math.max(1, ix2 - ix1);
                        const itemH = Math.max(1, iy2 - iy1);
                        if ((interW * interH) / (itemW * itemH) > 0.15) {
                            hasCarriedHelmet = true;
                            break;
                        }
                    }
                }
            }
        }

        if (state.imageRules.helmet_arm && hasCarriedHelmet) {
            isCompliant = false;
            missingChips.push("Helmet in Hand / Arm");
        }
        if (state.imageRules.gloves && !w.detected_ppe.gloves) {
            isCompliant = false;
            missingChips.push("No gloves");
        }
        if (state.imageRules.boots && !w.detected_ppe.boots) {
            isCompliant = false;
            missingChips.push("No safety boots");
        }
        if (state.imageRules.goggles && !w.detected_ppe.goggles) {
            isCompliant = false;
            missingChips.push("No eye protection");
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
            <div style="padding:32px 12px; text-align:center; color:var(--text-disabled); font-size:12px;">
                Upload a photo or select a test sample to evaluate workers.
            </div>
        `;
        return;
    }

    container.innerHTML = workers.map(w => {
        const isSelected = state.selectedWorkerId === w.worker_id;
        const badgeClass = w.is_compliant ? "compliant" : "violation";
        const badgeText = w.is_compliant ? "COMPLIANT" : "VIOLATION";
        const confPct = Math.round(w.confidence * 100);

        let chipsHtml = "";
        if (w.is_compliant) {
            const hasH = !!(w.detected_ppe && w.detected_ppe.helmet);
            const hasV = !!(w.detected_ppe && w.detected_ppe.vest);
            if (hasH && hasV) {
                chipsHtml = `<span class="compliant-pill-chip">Full Helmet & Vest</span>`;
            } else if (hasH) {
                chipsHtml = `<span class="compliant-pill-chip">Helmet Worn</span>`;
            } else if (hasV) {
                chipsHtml = `<span class="compliant-pill-chip" style="background:rgba(234,179,8,0.12); color:#FACC15; border-color:rgba(234,179,8,0.3);">Vest Worn (No Helmet)</span>`;
            } else {
                chipsHtml = `<span class="compliant-pill-chip" style="background:#262626; color:#A3A3A3; border-color:#404040;">No PPE (Rules Inactive)</span>`;
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
                    <span>Photo Review</span>
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
            const labelText = `Worker #${w.worker_id} - ${isViolation ? 'Violation' : 'Compliant'} - ${confPct}%`;
            
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
    metaText.innerText = `${file.name} • Processing frames through RF-DETR model on RTX 3050...`;

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

            metaText.innerText = `${file.name} • ${data.total_frames_processed} Frames Analyzed • ${data.cumulative_violations_spotted} Violations Flagged • ${(latency/1000).toFixed(1)}s`;

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
        const hasCarriedHelmet = (inc.negative_flags && inc.negative_flags.includes("helmet_under_arm")) ||
                                 (inc.missing_ppe && inc.missing_ppe.includes("helmet_under_arm"));
        if (state.videoRules.helmet_arm && hasCarriedHelmet) {
            isComp = false;
            missing.push("Helmet in Hand / Arm");
        }
        if (state.videoRules.gloves && !inc.detected_ppe?.gloves && "gloves" in (inc.detected_ppe || {})) {
            isComp = false;
            missing.push("No gloves");
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
            <div style="padding:32px 12px; text-align:center; color:var(--text-disabled); font-size:12px;">
                Upload a video to audit compliance across frames.
            </div>
        `;
        return;
    }

    if (badge) badge.innerText = `${incidents.length} Incident${incidents.length !== 1 ? 's' : ''}`;

    container.innerHTML = incidents.map((inc, idx) => {
        const isComp = inc.is_compliant;
        const badgeClass = isComp ? "compliant" : "violation";
        const badgeText = isComp ? "COMPLIANT" : "VIOLATION";
        const confPct = Math.round(inc.confidence * 100);
        const revStatus = inc.reviewer_status || "confirm";

        const chipsHtml = isComp
            ? `<span class="compliant-pill-chip">Full Helmet & Vest</span>`
            : (inc.missing_chips || []).map(chip => `<span class="violation-pill-chip">${chip}</span>`).join("");

        return `
            <div class="worker-verdict-card" onclick="seekVideoToTime(${inc.timestamp_sec})" title="Click to seek video to ${inc.timestamp_sec}s">
                <div class="card-top-row">
                    <div class="worker-meta-left">
                        <span class="worker-id-badge">Frame #${inc.frame} (${inc.timestamp_sec}s)</span>
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
        if (!video || !video.videoWidth) return null;

        const off = document.createElement("canvas");
        const scale = Math.min(1.0, 960 / Math.max(video.videoWidth, video.videoHeight));
        off.width = Math.round(video.videoWidth * scale);
        off.height = Math.round(video.videoHeight * scale);
        const ctx = off.getContext("2d");

        // Composite live video frame and AI bounding box overlay
        ctx.drawImage(video, 0, 0, off.width, off.height);
        if (canvas) {
            ctx.drawImage(canvas, 0, 0, off.width, off.height);
        }
        return off.toDataURL("image/jpeg", 0.80);
    } catch (e) {
        console.warn("Could not capture camera snapshot:", e);
        return null;
    }
}

function captureImageInspectionSnapshot() {
    // 1. Prefer backend processed image URL if available (persistent on server disk)
    if (state.currentImageDetections && state.currentImageDetections.processed_image_url) {
        return state.currentImageDetections.processed_image_url;
    }
    if (state.currentImageDetections && state.currentImageDetections.annotated_image_base64) {
        return state.currentImageDetections.annotated_image_base64;
    }
    // 2. Offscreen composite using correct element IDs: #image-display-element and #image-overlay-canvas
    try {
        const img = document.getElementById("image-display-element");
        const canvas = document.getElementById("image-overlay-canvas");
        if (!img || !img.naturalWidth) return null;

        const off = document.createElement("canvas");
        const scale = Math.min(1.0, 960 / Math.max(img.naturalWidth, img.naturalHeight));
        off.width = Math.round(img.naturalWidth * scale);
        off.height = Math.round(img.naturalHeight * scale);
        const ctx = off.getContext("2d");

        ctx.drawImage(img, 0, 0, off.width, off.height);
        if (canvas) {
            ctx.drawImage(canvas, 0, 0, off.width, off.height);
        }
        return off.toDataURL("image/jpeg", 0.80);
    } catch (e) {
        console.warn("Could not capture image inspection snapshot:", e);
        return null;
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

    if (isVideo) {
        summary = violations.slice(0, 3).map(i => `Frame #${i.frame}: ${(i.missing_chips || []).join(', ')}`).join(" | ");
        if (violations.length > 3) summary += ` (+${violations.length - 3} more)`;
        snapshotImage = captureVideoInspectionSnapshot();
    } else {
        summary = violations.map(w => `Worker #${w.worker_id}: ${(w.missing_chips || []).join(', ')}`).join(" | ");
        snapshotImage = captureImageInspectionSnapshot();
    }

    const logEntry = {
        id: `INC-${Date.now().toString().slice(-6)}`,
        timestamp: new Date().toLocaleString(),
        camera: sourceName,
        totalWorkers: count,
        compliant: Math.max(0, count - violations.length),
        violations: violations.length,
        verdict: summary,
        snapshotImage: snapshotImage
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
            snapshotImage = captureCameraSnapshot(targetCam);
        }
    }

    const logEntry = {
        id: `INC-${Date.now().toString().slice(-6)}`,
        timestamp: new Date().toLocaleString(),
        camera: sourceName,
        totalWorkers: count,
        compliant: Math.max(0, count - violCount),
        violations: violCount,
        verdict: summary,
        snapshotImage: snapshotImage
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
        console.error("Failed to save audit logs to localStorage", e);
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
                <td colspan="9" style="text-align:center; color:var(--text-disabled); padding:32px;">
                    No audit log incidents recorded yet. Perform inspections in CCTV, Image, or Video tabs and click 'Add to Audit Report'.
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = state.auditLogs.map((log, idx) => {
        const isViol = log.violations > 0;
        const badgeClass = isViol ? "violation" : "compliant";
        const badgeText = isViol ? `${log.violations} VIOLATION${log.violations > 1 ? 'S' : ''}` : "COMPLIANT";

        const proofThumb = log.snapshotImage 
            ? `<div class="audit-snapshot-thumb" ondblclick="openProofPage('${log.id}')" onclick="openProofPage('${log.id}')" title="Double-click or click to open full proof page in new tab">
                 <img src="${log.snapshotImage}" alt="Proof" />
                 <span class="thumb-zoom-icon"><i data-lucide="external-link" style="width:12px; height:12px;"></i></span>
               </div>`
            : `<div class="audit-snapshot-thumb" ondblclick="openProofPage('${log.id}')" onclick="openProofPage('${log.id}')" style="cursor:pointer;" title="Click to view incident details">
                 <span style="font-size:10px; color:var(--text-disabled); font-style:italic; text-align:center;">View<br>Proof</span>
               </div>`;

        return `
            <tr>
                <td style="font-family:monospace; font-weight:700; color:var(--text-primary); cursor:pointer;" onclick="openProofPage('${log.id}')" title="Open full incident record">${log.id}</td>
                <td>${proofThumb}</td>
                <td style="font-size:12px; color:var(--text-secondary);">${log.timestamp}</td>
                <td style="font-weight:600;">${log.camera}</td>
                <td>${log.totalWorkers}</td>
                <td style="color:var(--compliant-text); font-weight:600;">${log.compliant}</td>
                <td style="color:${isViol ? 'var(--accent)' : 'var(--text-secondary)'}; font-weight:700;">${log.violations}</td>
                <td>
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span class="status-badge-chip ${badgeClass}">${badgeText}</span>
                        <span style="font-size:12px; color:var(--text-secondary);">${log.verdict}</span>
                    </div>
                </td>
                <td>
                    <div style="display:flex; gap:6px;">
                        <button class="btn-outlined" style="padding:2px 8px; font-size:11px;" onclick="openProofPage('${log.id}')" title="Open full proof page in new tab">
                            Proof
                        </button>
                        <button class="btn-outlined" style="padding:2px 8px; font-size:11px;" onclick="deleteAuditLogEntry(${idx})">
                            Delete
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
    const log = (state.auditLogs || []).find(l => l.id === logId);
    if (!log || !log.snapshotImage) return;

    currentLightboxLogId = logId;
    const modal = document.getElementById("proof-lightbox-modal");
    const img = document.getElementById("lightbox-image");
    const idEl = document.getElementById("lightbox-incident-id");
    const metaEl = document.getElementById("lightbox-incident-meta");
    const verdictEl = document.getElementById("lightbox-verdict-text");
    const timeEl = document.getElementById("lightbox-timestamp-text");

    if (idEl) idEl.innerText = log.id;
    if (metaEl) metaEl.innerText = `${log.camera} • ${log.violations > 0 ? log.violations + ' Violation(s)' : 'Compliant'}`;
    if (img) img.src = log.snapshotImage;
    if (verdictEl) verdictEl.innerText = log.verdict;
    if (timeEl) timeEl.innerText = `Recorded: ${log.timestamp} | Total Workers: ${log.totalWorkers} (Compliant: ${log.compliant})`;

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
