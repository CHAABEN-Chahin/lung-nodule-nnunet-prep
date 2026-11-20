import { ChevronRight, FileText, Home, LogOut, Monitor, Plus, Settings, User, FileBarChart, Loader2, X, FolderOpen, Trash2, Play } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { useState, useEffect, useCallback } from "react";
import nodulisLogo from "../../imports/Nodulis1.png";
import { API, getAuthHeaders } from "../App";
import type { UserInfo } from "../App";

interface DashboardProps {
  onNavigate: (screen: string, caseId?: string) => void;
  onLogout: () => void;
  user: UserInfo | null;
}

interface CaseItem {
  id: number;
  case_id: string;
  patient_name: string;
  n_dossier: string;
  scan_date: string;
  created_at: string;
  status: string;
  dicom_path: string;
  num_nodules: number;
  error_message: string | null;
}

export function Dashboard({ onNavigate, onLogout, user }: DashboardProps) {
  const [activeSection, setActiveSection] = useState("home");
  const [cases, setCases] = useState<CaseItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [showNewCase, setShowNewCase] = useState(false);

  const fetchCases = useCallback(() => {
    fetch(`${API}/cases`, { headers: getAuthHeaders() })
      .then((r) => r.json())
      .then((data) => { setCases(data); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchCases();
    const interval = setInterval(fetchCases, 5000);
    return () => clearInterval(interval);
  }, [fetchCases]);

  const handleDelete = async (caseId: string) => {
    await fetch(`${API}/cases/${caseId}`, { method: "DELETE", headers: getAuthHeaders() });
    fetchCases();
  };

  const handleAnalyze = async (caseId: string) => {
    await fetch(`${API}/cases/${caseId}/analyze`, { method: "POST", headers: getAuthHeaders() });
    fetchCases();
  };

  const handleNavigation = (section: string, screen: string, caseId?: string) => {
    setActiveSection(section);
    onNavigate(screen, caseId);
  };

  return (
    <div className="min-h-screen bg-[#f8f9fb] flex">
      {/* Sidebar */}
      <motion.aside initial={{ x: -20, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ duration: 0.4 }}
        className="w-[260px] bg-white border-r border-gray-200 flex flex-col">
        <div className="h-[72px] flex items-center px-6 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-white border border-gray-200 flex items-center justify-center p-1.5">
              <img src={nodulisLogo} alt="Nodulis" className="w-full h-full object-contain" />
            </div>
            <span className="text-[17px] tracking-tight text-gray-900">Nodulis</span>
          </div>
        </div>
        <nav className="flex-1 px-3 py-6 space-y-1">
          {[
            { id: "home", icon: Home, label: "Home", action: () => setActiveSection("home") },
            { id: "cases", icon: FileText, label: "Cases", action: () => setActiveSection("cases") },
            { id: "settings", icon: Settings, label: "Settings", action: () => handleNavigation("settings", "settings") },
          ].map((item) => (
            <button key={item.id} onClick={item.action}
              className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-colors ${activeSection === item.id ? "bg-[#4b73a7]/10 text-[#4b73a7]" : "text-gray-700 hover:bg-gray-100"}`}>
              <item.icon className="w-[18px] h-[18px]" />
              <span className="text-[14px]">{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="p-3 border-t border-gray-200">
          <div className="flex items-center gap-3 px-3 py-2 mb-2">
            <div className="w-8 h-8 rounded-lg bg-[#c6ddf7] flex items-center justify-center">
              <User className="w-[16px] h-[16px] text-[#4b73a7]" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-[13px] text-gray-900 truncate">{user?.full_name || "Doctor"}</p>
              <p className="text-[11px] text-gray-500">{user?.department || ""}</p>
            </div>
          </div>
          <button onClick={onLogout} className="w-full flex items-center gap-3 px-4 py-2 rounded-lg text-gray-700 hover:bg-gray-100 transition-colors">
            <LogOut className="w-[16px] h-[16px]" />
            <span className="text-[13px]">Logout</span>
          </button>
        </div>
      </motion.aside>

      {/* Main */}
      <div className="flex-1 flex flex-col">
        <motion.header initial={{ y: -10, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ duration: 0.4, delay: 0.1 }}
          className="h-[72px] bg-white border-b border-gray-200 flex items-center justify-between px-8">
          <div>
            <h1 className="text-[20px] text-gray-900">Home</h1>
            <p className="text-[13px] text-gray-500 mt-0.5">Manage and review cases</p>
          </div>
          <div className="text-right">
            <p className="text-[12px] text-gray-500">Today</p>
            <p className="text-[13px] text-gray-900">{new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}</p>
          </div>
        </motion.header>

        <motion.main initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ duration: 0.5, delay: 0.2 }}
          className="flex-1 p-8 overflow-auto">
          <div className="max-w-[1400px] mx-auto space-y-8">
            {/* New Case Button */}
            <button onClick={() => setShowNewCase(true)}
              className="w-full bg-gradient-to-br from-[#4b73a7] to-[#3d5f8a] hover:from-[#3d5f8a] hover:to-[#2f4a6f] text-white rounded-2xl p-8 flex items-center justify-between transition-all shadow-lg shadow-[#4b73a7]/20 hover:shadow-[#4b73a7]/30">
              <div className="text-left">
                <h2 className="text-[24px] mb-2">New Case</h2>
                <p className="text-[14px] text-[#c6ddf7]">Import and analyze CT scan data</p>
              </div>
              <div className="w-14 h-14 rounded-xl bg-white/10 backdrop-blur flex items-center justify-center">
                <Plus className="w-7 h-7" />
              </div>
            </button>

            {/* Cases table */}
            <div>
              <h3 className="text-[18px] text-gray-900 mb-4">
                {cases.length > 0 ? `All Cases (${cases.length})` : "No Cases Yet"}
              </h3>
              {loading ? (
                <div className="flex items-center gap-2 text-gray-500 py-8 justify-center">
                  <Loader2 className="w-5 h-5 animate-spin" /><span>Loading cases...</span>
                </div>
              ) : cases.length > 0 ? (
                <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
                  <div className="grid grid-cols-[1.2fr_1fr_1fr_0.8fr_0.8fr_1fr_auto] gap-4 px-6 py-4 bg-[#c6ddf7]/20 border-b border-gray-200">
                    {["Case ID", "Patient", "N. Dossier", "Date", "Nodules", "Status", ""].map((h) => (
                      <div key={h} className="text-[12px] text-gray-700 uppercase tracking-wide">{h}</div>
                    ))}
                  </div>
                  {cases.map((c, index) => (
                    <motion.div key={c.case_id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 + index * 0.03 }}
                      className="grid grid-cols-[1.2fr_1fr_1fr_0.8fr_0.8fr_1fr_auto] gap-4 px-6 py-4 border-b border-gray-100 last:border-b-0 hover:bg-[#c6ddf7]/10 transition-colors">
                      <div className="text-[14px] text-gray-900 font-medium">{c.case_id}</div>
                      <div className="text-[14px] text-gray-700">{c.patient_name || "—"}</div>
                      <div className="text-[14px] text-gray-600">{c.n_dossier || "—"}</div>
                      <div className="text-[14px] text-gray-600">{c.scan_date}</div>
                      <div className="text-[14px] text-gray-700">{c.status === "complete" ? c.num_nodules : "—"}</div>
                      <div>
                        <span className={`inline-flex items-center px-2.5 py-1 rounded-md text-[12px] ${
                          c.status === "complete" ? "bg-green-50 text-green-700" :
                          c.status === "processing" ? "bg-blue-50 text-blue-700" :
                          c.status === "failed" ? "bg-red-50 text-red-700" :
                          "bg-amber-50 text-amber-700"
                        }`}>
                          {c.status === "processing" && <Loader2 className="w-3 h-3 animate-spin mr-1" />}
                          {c.status.charAt(0).toUpperCase() + c.status.slice(1)}
                        </span>
                      </div>
                      <div className="flex items-center gap-1">
                        {c.status === "uploaded" && (
                          <button onClick={() => handleAnalyze(c.case_id)} title="Run Analysis"
                            className="p-2 rounded-lg text-[#4b73a7] hover:bg-[#4b73a7]/10 transition-colors">
                            <Play className="w-4 h-4" />
                          </button>
                        )}
                        {c.status === "complete" && (
                          <button onClick={() => handleNavigation("visualization", "visualization", c.case_id)}
                            className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg text-[13px] text-[#4b73a7] hover:bg-[#4b73a7]/10 transition-colors">
                            Open<ChevronRight className="w-3.5 h-3.5" />
                          </button>
                        )}
                        <button onClick={() => handleDelete(c.case_id)} title="Delete"
                          className="p-2 rounded-lg text-gray-400 hover:text-red-500 hover:bg-red-50 transition-colors">
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </div>
              ) : (
                <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
                  <FolderOpen className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500 text-[14px]">No cases yet. Click "New Case" to get started.</p>
                </div>
              )}
            </div>
          </div>
        </motion.main>
      </div>

      {/* New Case Modal */}
      <AnimatePresence>
        {showNewCase && <NewCaseModal onClose={() => setShowNewCase(false)} onCreated={(caseId) => { setShowNewCase(false); fetchCases(); handleAnalyze(caseId); }} />}
      </AnimatePresence>
    </div>
  );
}

function NewCaseModal({ onClose, onCreated }: { onClose: () => void; onCreated: (caseId: string) => void }) {
  const [patientName, setPatientName] = useState("");
  const [nDossier, setNDossier] = useState("");
  const [scanDate, setScanDate] = useState(new Date().toLocaleDateString("fr-FR"));
  const [dicomPath, setDicomPath] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!dicomPath.trim()) { setError("DICOM folder path is required"); return; }
    setError("");
    setLoading(true);
    try {
      const res = await fetch(`${API}/cases`, {
        method: "POST",
        headers: getAuthHeaders(),
        body: JSON.stringify({ patient_name: patientName, n_dossier: nDossier, scan_date: scanDate, dicom_path: dicomPath }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.error || "Failed to create case"); return; }
      onCreated(data.case_id);
    } catch {
      setError("Failed to connect to backend");
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
      <motion.div initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }}
        className="bg-white rounded-2xl w-[500px] shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between px-6 py-5 border-b border-gray-200">
          <h2 className="text-[18px] text-gray-900">New Case</h2>
          <button onClick={onClose} className="p-1 rounded-lg hover:bg-gray-100"><X className="w-5 h-5 text-gray-500" /></button>
        </div>
        <form onSubmit={handleSubmit} className="p-6 space-y-5">
          {error && <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-3"><p className="text-[13px] text-red-600">{error}</p></div>}
          <div>
            <label className="block text-[13px] text-gray-700 mb-2">Patient Name</label>
            <input type="text" value={patientName} onChange={(e) => setPatientName(e.target.value)} placeholder="e.g. Mohamed Ben Ali"
              className="w-full px-4 py-2.5 rounded-lg border border-gray-300 text-[14px] focus:outline-none focus:border-[#4b73a7] focus:ring-2 focus:ring-[#4b73a7]/20" />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-[13px] text-gray-700 mb-2">N. Dossier</label>
              <input type="text" value={nDossier} onChange={(e) => setNDossier(e.target.value)} placeholder="e.g. 207/2026"
                className="w-full px-4 py-2.5 rounded-lg border border-gray-300 text-[14px] focus:outline-none focus:border-[#4b73a7] focus:ring-2 focus:ring-[#4b73a7]/20" />
            </div>
            <div>
              <label className="block text-[13px] text-gray-700 mb-2">Scan Date</label>
              <input type="text" value={scanDate} onChange={(e) => setScanDate(e.target.value)}
                className="w-full px-4 py-2.5 rounded-lg border border-gray-300 text-[14px] focus:outline-none focus:border-[#4b73a7] focus:ring-2 focus:ring-[#4b73a7]/20" />
            </div>
          </div>
          <div>
            <label className="block text-[13px] text-gray-700 mb-2">DICOM Folder <span className="text-red-500">*</span></label>
            <div className="flex gap-2">
              <div className="relative flex-1">
                <FolderOpen className="absolute left-3 top-1/2 -translate-y-1/2 w-[18px] h-[18px] text-gray-400" />
                <input type="text" value={dicomPath} readOnly
                  placeholder="No folder selected"
                  className="w-full pl-10 pr-4 py-2.5 rounded-lg border border-gray-300 text-[14px] focus:outline-none focus:border-[#4b73a7] focus:ring-2 focus:ring-[#4b73a7]/20 font-mono text-[13px] bg-gray-50 cursor-default" />
              </div>
              <button type="button" onClick={async () => {
                try {
                  const res = await fetch(`${API}/browse-folder`, { method: "POST", headers: getAuthHeaders() });
                  const data = await res.json();
                  if (data.path) setDicomPath(data.path);
                } catch { /* ignore */ }
              }}
                className="px-4 py-2.5 rounded-lg border border-gray-300 text-[14px] text-gray-700 hover:bg-gray-100 transition-colors flex items-center gap-2 whitespace-nowrap">
                <FolderOpen className="w-4 h-4" />
                Browse
              </button>
            </div>
            <p className="text-[11px] text-gray-500 mt-1.5">Select the folder containing .dcm files on this machine</p>
          </div>
          <div className="flex justify-end gap-3 pt-2">
            <button type="button" onClick={onClose} className="px-5 py-2.5 rounded-lg border border-gray-300 text-[14px] text-gray-700 hover:bg-gray-50 transition-colors">Cancel</button>
            <button type="submit" disabled={loading}
              className="px-5 py-2.5 rounded-lg bg-[#4b73a7] hover:bg-[#3d5f8a] text-white text-[14px] transition-colors disabled:opacity-60 flex items-center gap-2">
              {loading && <Loader2 className="w-4 h-4 animate-spin" />}
              Create & Analyze
            </button>
          </div>
        </form>
      </motion.div>
    </motion.div>
  );
}
