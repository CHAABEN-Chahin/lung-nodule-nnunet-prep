import { ArrowLeft, Download, Eye, Monitor, Loader2, Layers, Activity, Target } from "lucide-react";
import { motion } from "motion/react";
import { useState, useEffect, useRef } from "react";
import { API, getAuthHeaders } from "../App";

interface VisualizationScreenProps {
  onNavigate: (screen: string) => void;
  caseId: string;
}

type ViewTarget = "nodule" | "lungs" | "lobes" | "all-nodules";

function riskLevel(diam: number): { label: string; color: string; bg: string } {
  if (diam >= 8) return { label: "High", color: "text-[#ff9999]", bg: "bg-[#850808]/30" };
  if (diam >= 6) return { label: "Medium", color: "text-amber-400", bg: "bg-amber-500/20" };
  return { label: "Low", color: "text-green-400", bg: "bg-green-500/20" };
}

interface Nodule {
  nodule_id: string;
  diameter_mm: number;
  volume_mm3: number;
  centroid_z_vox: number;
  centroid_y_vox: number;
  centroid_x_vox: number;
  sphericity: number;
  hu_mean: number;
  lobe?: string;
  [key: string]: unknown;
}

interface LobeInfo {
  label: number;
  name: string;
  side: string;
  voxels: number;
}

const LOBE_COLORS: Record<number, string> = {
  1: "#3B82F6", 2: "#06B6D4", 3: "#F59E0B", 4: "#10B981", 5: "#8B5CF6",
  10: "#3B82F6", 11: "#06B6D4", 12: "#F59E0B", 13: "#10B981", 14: "#8B5CF6",
};

function lobeColor(label: number): string {
  return LOBE_COLORS[label] || "#6B7280";
}

function NiivueViewer({ caseId, noduleId, viewTarget, viewMode, selectedLobe, slicePlane }: {
  caseId: string;
  noduleId: string | null;
  viewTarget: ViewTarget;
  viewMode: "2D" | "3D";
  selectedLobe: number | null;
  slicePlane: "Multi" | "A" | "S" | "C";
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nvRef = useRef<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const keyRef = useRef("");

  useEffect(() => {
    const newKey = viewTarget === "lobes"
      ? `lobes-${selectedLobe}`
      : viewTarget === "nodule"
        ? `nodule-${noduleId}`
        : viewTarget;
    if (newKey === keyRef.current && nvRef.current) return;
    keyRef.current = newKey;
    if (!canvasRef.current) return;

    setLoading(true);
    setError(null);

    const token = localStorage.getItem("nodulis_token") || "";

    const loadNiivue = async () => {
      try {
        if (!(window as any).niivue) {
          await new Promise<void>((resolve, reject) => {
            const s = document.createElement("script");
            s.src = "https://cdn.jsdelivr.net/npm/@niivue/niivue@0.48.0/dist/niivue.umd.min.js";
            s.onload = () => resolve();
            s.onerror = () => reject(new Error("Failed to load Niivue"));
            document.head.appendChild(s);
          });
        }

        const { Niivue } = (window as any).niivue;
        const nv = new Niivue({
          backColor: [0.05, 0.05, 0.08, 1],
          crosshairColor: [0.29, 0.45, 0.65, 1],
          show3Dcrosshair: viewTarget === "nodule",
        });
        await nv.attachToCanvas(canvasRef.current);

        let volumes: any[] = [];
        const hdr = { Authorization: `Bearer ${token}` };

        if (viewTarget === "nodule" && noduleId) {
          volumes = [
            { url: `${API}/cases/${caseId}/nodules/${noduleId}/patch`, name: "patch.nii.gz", colormap: "gray", opacity: 1, headers: hdr },
            { url: `${API}/cases/${caseId}/nodules/${noduleId}/mask`, name: "mask.nii.gz", colormap: "warm", opacity: 0.5, headers: hdr },
          ];
        } else if (viewTarget === "lungs") {
          volumes = [
            { url: `${API}/cases/${caseId}/volumes/ct-lungs`, name: "ct_lungs.nii.gz", colormap: "winter", opacity: 0.8, headers: hdr },
          ];
        } else if (viewTarget === "lobes" && selectedLobe !== null) {
          volumes = [
            { url: `${API}/cases/${caseId}/volumes/ct-lobe/${selectedLobe}`, name: `lobe_${selectedLobe}.nii.gz`, colormap: "winter", opacity: 0.8, headers: hdr },
          ];
        } else if (viewTarget === "all-nodules") {
          volumes = [
            { url: `${API}/cases/${caseId}/volumes/ct-lungs`, name: "ct_lungs.nii.gz", colormap: "winter", opacity: 0.35, headers: hdr },
            { url: `${API}/cases/${caseId}/volumes/nodule-mask`, name: "nodules.nii.gz", colormap: "red", opacity: 1, headers: hdr },
          ];
        }

        if (volumes.length === 0) {
          setError("Select a lobe to visualize.");
          setLoading(false);
          return;
        }

        try {
          await nv.loadVolumes(volumes);
        } catch {
          setError("Volume files not available.\nRun the analysis pipeline first.");
          setLoading(false);
          return;
        }

        if (viewTarget === "lungs" || viewTarget === "lobes") {
          nv.volumes[0].cal_min = -600;
          nv.volumes[0].cal_max = 150;
          nv.updateGLVolume();
        }
        if (viewTarget === "all-nodules") {
          nv.volumes[0].cal_min = -600;
          nv.volumes[0].cal_max = 150;
          nv.updateGLVolume();
        }

        if (viewMode === "3D") {
          nv.setSliceType(nv.sliceTypeRender);
        } else {
          applySlicePlane(nv, slicePlane);
        }

        nvRef.current = nv;
        setLoading(false);
      } catch {
        setError("Could not initialize 3D viewer.");
        setLoading(false);
      }
    };
    loadNiivue();
    return () => { nvRef.current = null; };
  }, [caseId, noduleId, viewTarget, selectedLobe]);

  useEffect(() => {
    if (!nvRef.current) return;
    const nv = nvRef.current as any;
    if (viewMode === "3D") {
      nv.setSliceType(nv.sliceTypeRender);
    } else {
      applySlicePlane(nv, slicePlane);
    }
  }, [viewMode, slicePlane]);

  return (
    <div className="relative w-full h-full">
      {loading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/60 z-10">
          <Loader2 className="w-8 h-8 text-[#4b73a7] animate-spin" />
          <span className="text-gray-400 text-[13px]">Loading volume...</span>
        </div>
      )}
      {error && !loading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-[#08090e]">
          <Monitor className="w-16 h-16 text-gray-700" />
          <p className="text-[12px] text-gray-500 text-center whitespace-pre-line">{error}</p>
        </div>
      )}
      <canvas ref={canvasRef} className="w-full h-full" style={{ display: error ? "none" : "block" }} />
    </div>
  );
}

function applySlicePlane(nv: any, plane: "Multi" | "A" | "S" | "C") {
  if (plane === "A") nv.setSliceType(nv.sliceTypeAxial);
  else if (plane === "S") nv.setSliceType(nv.sliceTypeSagittal);
  else if (plane === "C") nv.setSliceType(nv.sliceTypeCoronal);
  else nv.setSliceType(nv.sliceTypeMultiplanar);
}

export function VisualizationScreen({ onNavigate, caseId }: VisualizationScreenProps) {
  const [selectedNodule, setSelectedNodule] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"2D" | "3D">("3D");
  const [viewTarget, setViewTarget] = useState<ViewTarget>("lungs");
  const [slicePlane, setSlicePlane] = useState<"Multi" | "A" | "S" | "C">("Multi");
  const [nodules, setNodules] = useState<Nodule[]>([]);
  const [loading, setLoading] = useState(true);
  const [caseInfo, setCaseInfo] = useState<any>(null);
  const [lobeInfo, setLobeInfo] = useState<LobeInfo[]>([]);
  const [selectedLobe, setSelectedLobe] = useState<number | null>(null);

  useEffect(() => {
    fetch(`${API}/cases/${caseId}/nodules`, { headers: getAuthHeaders() })
      .then((r) => r.json())
      .then((data: Nodule[]) => { setNodules(data); setLoading(false); })
      .catch(() => setLoading(false));

    fetch(`${API}/cases/${caseId}`, { headers: getAuthHeaders() })
      .then((r) => r.json())
      .then(setCaseInfo)
      .catch(() => {});

    fetch(`${API}/cases/${caseId}/lobe-info`, { headers: getAuthHeaders() })
      .then((r) => r.ok ? r.json() : [])
      .then((data: LobeInfo[]) => {
        setLobeInfo(data);
        if (data.length > 0 && selectedLobe === null) setSelectedLobe(data[0].label);
      })
      .catch(() => {});
  }, [caseId]);

  const selected = nodules.find((n) => n.nodule_id === selectedNodule);

  const leftLobes = lobeInfo.filter(l => l.side === "left");
  const rightLobes = lobeInfo.filter(l => l.side === "right");

  const viewTargetLabel: Record<ViewTarget, string> = {
    lungs: "Lungs — CT Interior View",
    lobes: selectedLobe !== null
      ? `${lobeInfo.find(l => l.label === selectedLobe)?.name ?? "Lobe"} — CT Interior View`
      : "Select a Lobe",
    "all-nodules": "All Detected Nodules",
    nodule: selectedNodule
      ? selectedNodule.replace("_", " ").replace(/\b\w/g, (c) => c.toUpperCase())
      : "Select a nodule",
  };

  return (
    <div className="min-h-screen bg-[#0a0d14] text-white flex flex-col">
      {/* Header */}
      <motion.header initial={{ y: -20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ duration: 0.4 }}
        className="h-[60px] bg-[#0f1319] border-b border-gray-800 flex items-center justify-between px-6">
        <div className="flex items-center gap-6">
          <button onClick={() => onNavigate("dashboard")} className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors">
            <ArrowLeft className="w-[18px] h-[18px]" />
            <span className="text-[14px]">Back</span>
          </button>
          <div className="h-5 w-px bg-gray-800" />
          <h1 className="text-[15px] text-gray-300">Case {caseId}</h1>
          {caseInfo?.patient_name && <span className="text-[13px] text-gray-500">— {caseInfo.patient_name}</span>}
        </div>
        <div className="flex items-center gap-2">
          {/* 2D / 3D toggle */}
          <div className="flex items-center gap-1 bg-gray-800 rounded-lg p-1">
            {(["2D", "3D"] as const).map((m) => (
              <button key={m} onClick={() => setViewMode(m)}
                className={`px-3 py-1.5 rounded-md text-[13px] transition-colors ${viewMode === m ? "bg-[#4b73a7] text-white" : "text-gray-400 hover:text-white"}`}>
                {m} View
              </button>
            ))}
          </div>

          {/* Slice plane selector — 2D mode only */}
          {viewMode === "2D" && (
            <>
              <div className="h-8 w-px bg-gray-800 mx-1" />
              <div className="flex items-center gap-1 bg-gray-800 rounded-lg p-1">
                {([
                  { id: "Multi" as const, tip: "Multi-planar" },
                  { id: "A" as const, tip: "Axial" },
                  { id: "S" as const, tip: "Sagittal" },
                  { id: "C" as const, tip: "Coronal" },
                ]).map((p) => (
                  <button key={p.id} onClick={() => setSlicePlane(p.id)} title={p.tip}
                    className={`px-2.5 py-1.5 rounded-md text-[12px] transition-colors ${slicePlane === p.id ? "bg-[#4b73a7] text-white" : "text-gray-400 hover:text-white"}`}>
                    {p.id === "Multi" ? "Multi" : p.id}
                  </button>
                ))}
              </div>
            </>
          )}

          <div className="h-8 w-px bg-gray-800 mx-1" />
          <button onClick={() => onNavigate("report")}
            className="px-4 py-2 rounded-lg bg-[#4b73a7] hover:bg-[#3d5f8a] transition-colors flex items-center gap-2">
            <Download className="w-[16px] h-[16px]" />
            <span className="text-[13px]">Report</span>
          </button>
        </div>
      </motion.header>

      <div className="flex-1 flex min-h-0">
        {/* Left panel */}
        <motion.aside initial={{ x: -20, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ duration: 0.4, delay: 0.1 }}
          className="w-[280px] bg-[#0f1319] border-r border-gray-800 flex flex-col overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* View Targets */}
            <div>
              <h3 className="text-[11px] text-gray-500 uppercase tracking-wide mb-3">Visualization</h3>
              <div className="space-y-1.5">
                {([
                  { id: "lungs" as ViewTarget, icon: Activity, label: "Lungs" },
                  { id: "lobes" as ViewTarget, icon: Layers, label: "5 Lobes" },
                  { id: "all-nodules" as ViewTarget, icon: Target, label: "All Nodules" },
                ] as const).map((item) => (
                  <button key={item.id}
                    onClick={() => {
                      setViewTarget(item.id);
                      setSelectedNodule(null);
                      if (item.id === "lobes" && lobeInfo.length > 0 && selectedLobe === null) {
                        setSelectedLobe(lobeInfo[0].label);
                      }
                    }}
                    className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors text-left ${
                      viewTarget === item.id && !selectedNodule ? "bg-[#4b73a7]/15 border border-[#4b73a7]/40" : "bg-[#14181f] border border-gray-800 hover:border-gray-700"
                    }`}>
                    <item.icon className="w-4 h-4 text-[#4b73a7]" />
                    <span className="text-[13px]">{item.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Lobe selector — visible when lobes view is active */}
            {viewTarget === "lobes" && lobeInfo.length > 0 && (
              <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} transition={{ duration: 0.3 }}>
                <h3 className="text-[11px] text-gray-500 uppercase tracking-wide mb-3">Select Lobe</h3>

                {leftLobes.length > 0 && (
                  <>
                    <p className="text-[10px] text-gray-600 uppercase tracking-wider mb-2 mt-2">Left Lung</p>
                    <div className="space-y-1">
                      {leftLobes.map(lobe => (
                        <button key={lobe.label}
                          onClick={() => setSelectedLobe(lobe.label)}
                          className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-all text-left text-[12px] ${
                            selectedLobe === lobe.label
                              ? "bg-[#4b73a7]/15 border border-[#4b73a7]/40 text-white"
                              : "bg-[#14181f] border border-gray-800 hover:border-gray-700 text-gray-400"
                          }`}>
                          <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: lobeColor(lobe.label) }} />
                          <span>{lobe.name}</span>
                        </button>
                      ))}
                    </div>
                  </>
                )}

                {rightLobes.length > 0 && (
                  <>
                    <p className="text-[10px] text-gray-600 uppercase tracking-wider mb-2 mt-3">Right Lung</p>
                    <div className="space-y-1">
                      {rightLobes.map(lobe => (
                        <button key={lobe.label}
                          onClick={() => setSelectedLobe(lobe.label)}
                          className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-all text-left text-[12px] ${
                            selectedLobe === lobe.label
                              ? "bg-[#4b73a7]/15 border border-[#4b73a7]/40 text-white"
                              : "bg-[#14181f] border border-gray-800 hover:border-gray-700 text-gray-400"
                          }`}>
                          <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: lobeColor(lobe.label) }} />
                          <span>{lobe.name}</span>
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </motion.div>
            )}

            {/* Case Info */}
            <div className="pt-4 border-t border-gray-800">
              <h3 className="text-[11px] text-gray-500 uppercase tracking-wide mb-3">Case Information</h3>
              <div className="space-y-3">
                <div><p className="text-[11px] text-gray-500 mb-1">Case ID</p><p className="text-[14px] text-white">{caseId}</p></div>
                {caseInfo?.patient_name && <div><p className="text-[11px] text-gray-500 mb-1">Patient</p><p className="text-[14px] text-white">{caseInfo.patient_name}</p></div>}
                {caseInfo?.n_dossier && <div><p className="text-[11px] text-gray-500 mb-1">N. Dossier</p><p className="text-[14px] text-white">{caseInfo.n_dossier}</p></div>}
                <div><p className="text-[11px] text-gray-500 mb-1">Nodules</p><p className="text-[14px] text-white">{loading ? "..." : `${nodules.length} nodule(s)`}</p></div>
                {selected && <>
                  <div><p className="text-[11px] text-gray-500 mb-1">Diameter</p><p className="text-[14px] text-white">{selected.diameter_mm.toFixed(1)} mm</p></div>
                  <div><p className="text-[11px] text-gray-500 mb-1">Volume</p><p className="text-[14px] text-white">{selected.volume_mm3.toFixed(1)} mm3</p></div>
                  <div><p className="text-[11px] text-gray-500 mb-1">HU Mean</p><p className="text-[14px] text-white">{selected.hu_mean.toFixed(1)}</p></div>
                </>}
              </div>
            </div>
          </div>
        </motion.aside>

        {/* Center viewer */}
        <motion.main initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ duration: 0.5, delay: 0.2 }}
          className="flex-1 flex flex-col bg-[#08090e] p-6">
          <div className="flex-1 flex flex-col">
            <div className="flex-1 bg-black rounded-xl overflow-hidden border border-gray-800 relative">
              <div className="absolute top-6 left-6 flex items-center gap-2 px-3 py-2 bg-black/60 backdrop-blur-sm rounded-lg border border-[#4b73a7]/30 z-20">
                <Monitor className="w-4 h-4 text-[#4b73a7]" />
                <span className="text-[12px] text-gray-300">{viewTargetLabel[viewTarget]}</span>
              </div>
              <NiivueViewer
                caseId={caseId}
                noduleId={selectedNodule}
                viewTarget={viewTarget}
                viewMode={viewMode}
                selectedLobe={selectedLobe}
                slicePlane={slicePlane}
              />
              {selected && (
                <div className="absolute bottom-6 right-6 px-3 py-2 bg-black/60 backdrop-blur-sm rounded-lg border border-gray-700 z-20">
                  <p className="text-[11px] text-gray-400 font-mono">
                    X: {selected.centroid_x_vox.toFixed(0)} Y: {selected.centroid_y_vox.toFixed(0)} Z: {selected.centroid_z_vox.toFixed(0)}
                  </p>
                </div>
              )}
            </div>
          </div>
        </motion.main>

        {/* Right panel — Nodule list */}
        <motion.aside initial={{ x: 20, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ duration: 0.4, delay: 0.1 }}
          className="w-[320px] bg-[#0f1319] border-l border-gray-800 flex flex-col overflow-y-auto">
          <div className="p-6 space-y-4">
            <h3 className="text-[11px] text-gray-500 uppercase tracking-wide">Detected Nodules</h3>
            {loading && (
              <div className="flex items-center gap-2 text-gray-500">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span className="text-[13px]">Loading...</span>
              </div>
            )}
            {!loading && nodules.length === 0 && <p className="text-[13px] text-gray-500">No nodules found.</p>}
            {nodules.map((nodule, index) => {
              const risk = riskLevel(nodule.diameter_mm);
              const isSelected = selectedNodule === nodule.nodule_id && viewTarget === "nodule";
              return (
                <motion.button key={nodule.nodule_id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 + index * 0.1 }}
                  onClick={() => { setSelectedNodule(nodule.nodule_id); setViewTarget("nodule"); }}
                  className={`w-full p-4 rounded-xl text-left transition-all ${isSelected ? "bg-[#4b73a7]/15 border border-[#4b73a7]/40" : "bg-[#14181f] border border-gray-800 hover:border-gray-700"}`}>
                  <div className="flex items-start justify-between mb-3">
                    <span className="text-[15px]">{nodule.nodule_id.replace("_", " ").replace(/\b\w/g, (c) => c.toUpperCase())}</span>
                    <span className={`text-[11px] px-2 py-0.5 rounded ${risk.bg} ${risk.color}`}>{risk.label}</span>
                  </div>
                  <div className="space-y-2 text-[12px]">
                    {([
                      ["Diameter", `${nodule.diameter_mm.toFixed(1)} mm`],
                      ["Volume", `${nodule.volume_mm3.toFixed(1)} mm3`],
                      ["Sphericity", nodule.sphericity?.toFixed(3) ?? "-"],
                      ["HU Mean", nodule.hu_mean.toFixed(1)],
                    ] as [string, string][]).map(([k, v]) => (
                      <div key={k} className="flex justify-between">
                        <span className="text-gray-500">{k}</span>
                        <span className="text-gray-300">{v}</span>
                      </div>
                    ))}
                  </div>
                  <div className={`w-full mt-3 px-3 py-2 rounded-lg text-[12px] transition-colors flex items-center justify-center gap-2 ${isSelected ? "bg-[#4b73a7] text-white" : "bg-gray-800 text-gray-400"}`}>
                    <Eye className="w-[14px] h-[14px]" />Focus View
                  </div>
                </motion.button>
              );
            })}
          </div>
        </motion.aside>
      </div>
    </div>
  );
}
