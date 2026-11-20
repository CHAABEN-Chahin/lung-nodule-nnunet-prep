import { ArrowLeft, Database, Lock, Monitor, Shield, User, Save, Loader2 } from "lucide-react";
import { motion } from "motion/react";
import { useState, useEffect } from "react";
import { API, getAuthHeaders } from "../App";
import type { UserInfo } from "../App";

interface SettingsScreenProps {
  onNavigate: (screen: string) => void;
  user: UserInfo | null;
  onUserUpdate: (user: UserInfo) => void;
}

export function SettingsScreen({ onNavigate, user, onUserUpdate }: SettingsScreenProps) {
  const [fullName, setFullName] = useState(user?.full_name || "");
  const [department, setDepartment] = useState(user?.department || "");
  const [email, setEmail] = useState(user?.email || "");
  const [hospitalName, setHospitalName] = useState(user?.hospital_name || "");
  const [service, setService] = useState(user?.service || "");
  const [newPassword, setNewPassword] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    fetch(`${API}/stats`, { headers: getAuthHeaders() })
      .then((r) => r.json())
      .then(setStats)
      .catch(() => {});
  }, []);

  const handleSave = async () => {
    setSaving(true);
    try {
      await fetch(`${API}/auth/update-profile`, {
        method: "PUT",
        headers: getAuthHeaders(),
        body: JSON.stringify({ full_name: fullName, department, email, hospital_name: hospitalName, service, ...(newPassword ? { new_password: newPassword } : {}) }),
      });
      onUserUpdate({ ...user!, full_name: fullName, department, email, hospital_name: hospitalName, service });
      setNewPassword("");
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f8f9fb] flex flex-col">
      <motion.header initial={{ y: -20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ duration: 0.4 }}
        className="h-[72px] bg-white border-b border-gray-200 flex items-center justify-between px-8">
        <div className="flex items-center gap-6">
          <button onClick={() => onNavigate("dashboard")} className="flex items-center gap-2 text-gray-600 hover:text-gray-900 transition-colors">
            <ArrowLeft className="w-[18px] h-[18px]" />
            <span className="text-[14px]">Back to Dashboard</span>
          </button>
          <div className="h-6 w-px bg-gray-300" />
          <div>
            <h1 className="text-[20px] text-gray-900">Settings</h1>
            <p className="text-[13px] text-gray-500">Application preferences and configuration</p>
          </div>
        </div>
        <button onClick={handleSave} disabled={saving}
          className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-[#4b73a7] hover:bg-[#3d5f8a] text-white transition-colors disabled:opacity-60">
          {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
          <span className="text-[14px]">{saved ? "Saved!" : "Save Changes"}</span>
        </button>
      </motion.header>

      <motion.main initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ duration: 0.5, delay: 0.1 }}
        className="flex-1 p-8 overflow-auto">
        <div className="max-w-[1000px] mx-auto space-y-6">
          {/* User Profile */}
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200 bg-[#c6ddf7]/20">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-[#4b73a7] flex items-center justify-center"><User className="w-5 h-5 text-white" /></div>
                <div><h2 className="text-[16px] text-gray-900">User Profile</h2><p className="text-[13px] text-gray-500">Your account settings (used in reports)</p></div>
              </div>
            </div>
            <div className="p-6 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-[13px] text-gray-700 mb-2">Full Name</label>
                  <input type="text" value={fullName} onChange={(e) => setFullName(e.target.value)}
                    className="w-full px-4 py-2.5 rounded-lg border border-gray-300 text-[14px] focus:outline-none focus:border-[#4b73a7] focus:ring-2 focus:ring-[#4b73a7]/20" />
                </div>
                <div>
                  <label className="block text-[13px] text-gray-700 mb-2">Department</label>
                  <input type="text" value={department} onChange={(e) => setDepartment(e.target.value)}
                    className="w-full px-4 py-2.5 rounded-lg border border-gray-300 text-[14px] focus:outline-none focus:border-[#4b73a7] focus:ring-2 focus:ring-[#4b73a7]/20" />
                </div>
              </div>
              <div>
                <label className="block text-[13px] text-gray-700 mb-2">Email</label>
                <input type="email" value={email} onChange={(e) => setEmail(e.target.value)}
                  className="w-full px-4 py-2.5 rounded-lg border border-gray-300 text-[14px] focus:outline-none focus:border-[#4b73a7] focus:ring-2 focus:ring-[#4b73a7]/20" />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-[13px] text-gray-700 mb-2">Hospital Name</label>
                  <input type="text" value={hospitalName} onChange={(e) => setHospitalName(e.target.value)}
                    className="w-full px-4 py-2.5 rounded-lg border border-gray-300 text-[14px] focus:outline-none focus:border-[#4b73a7] focus:ring-2 focus:ring-[#4b73a7]/20" />
                </div>
                <div>
                  <label className="block text-[13px] text-gray-700 mb-2">Service</label>
                  <input type="text" value={service} onChange={(e) => setService(e.target.value)}
                    className="w-full px-4 py-2.5 rounded-lg border border-gray-300 text-[14px] focus:outline-none focus:border-[#4b73a7] focus:ring-2 focus:ring-[#4b73a7]/20" />
                </div>
              </div>
            </div>
          </div>

          {/* Security */}
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200 bg-[#c6ddf7]/20">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-[#4b73a7] flex items-center justify-center"><Shield className="w-5 h-5 text-white" /></div>
                <div><h2 className="text-[16px] text-gray-900">Security</h2><p className="text-[13px] text-gray-500">Change your password</p></div>
              </div>
            </div>
            <div className="p-6">
              <div className="max-w-md">
                <label className="block text-[13px] text-gray-700 mb-2">New Password (leave blank to keep current)</label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} placeholder="Enter new password"
                    className="w-full pl-10 pr-4 py-2.5 rounded-lg border border-gray-300 text-[14px] focus:outline-none focus:border-[#4b73a7] focus:ring-2 focus:ring-[#4b73a7]/20" />
                </div>
              </div>
            </div>
          </div>

          {/* Data Management */}
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200 bg-[#c6ddf7]/20">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-[#4b73a7] flex items-center justify-center"><Database className="w-5 h-5 text-white" /></div>
                <div><h2 className="text-[16px] text-gray-900">Data Management</h2><p className="text-[13px] text-gray-500">Storage statistics</p></div>
              </div>
            </div>
            <div className="p-6">
              <div className="grid grid-cols-3 gap-4">
                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="text-[12px] text-gray-500 mb-1">Total Cases</p>
                  <p className="text-[20px] text-gray-900">{stats?.total_cases ?? "-"}</p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="text-[12px] text-gray-500 mb-1">Completed</p>
                  <p className="text-[20px] text-gray-900">{stats?.complete_cases ?? "-"}</p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="text-[12px] text-gray-500 mb-1">Total Nodules</p>
                  <p className="text-[20px] text-gray-900">{stats?.total_nodules ?? "-"}</p>
                </div>
              </div>
            </div>
          </div>

          {/* About */}
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <div className="text-center space-y-2">
              <p className="text-[14px] text-gray-900">Nodulis Platform</p>
              <p className="text-[12px] text-gray-500">Pulmonary Nodule Detection & Analysis</p>
              <p className="text-[11px] text-gray-400 pt-2">nnU-Net + CNN FP Reduction + PyRadiomics</p>
            </div>
          </div>
        </div>
      </motion.main>
    </div>
  );
}
