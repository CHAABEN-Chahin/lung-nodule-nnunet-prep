import { Lock, User, Loader2 } from "lucide-react";
import { motion } from "motion/react";
import { useState } from "react";
import nodulisLogo from "../../imports/Nodulis1.png";
import { API } from "../App";
import type { UserInfo } from "../App";

interface LoginScreenProps {
  onLogin: (token: string, user: UserInfo) => void;
}

export function LoginScreen({ onLogin }: LoginScreenProps) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const res = await fetch(`${API}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || "Login failed");
        return;
      }
      onLogin(data.token, data.user);
    } catch {
      setError("Cannot connect to backend. Make sure it is running on port 5000.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0e1a] relative overflow-hidden flex items-center justify-center">
      <div className="absolute inset-0 opacity-[0.03]">
        <div className="absolute inset-0" style={{ backgroundImage: `radial-gradient(circle at 2px 2px, white 1px, transparent 0)`, backgroundSize: "48px 48px" }} />
      </div>
      <div className="absolute inset-0 bg-gradient-to-br from-[#4b73a7]/20 via-transparent to-[#1a0f29] opacity-60" />

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }} className="relative z-10 w-full max-w-[420px] px-8">
        <div className="bg-[#151b2e]/80 backdrop-blur-xl rounded-2xl p-12 shadow-2xl border border-white/5">
          <div className="text-center mb-12">
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-xl bg-white/95 mb-6 shadow-lg shadow-[#4b73a7]/30 p-3">
              <img src={nodulisLogo} alt="Nodulis" className="w-full h-full object-contain" />
            </div>
            <h1 className="text-[28px] tracking-tight text-white mb-2">Nodulis</h1>
            <p className="text-[14px] text-gray-400">Lung Nodule Analysis Platform</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {error && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3">
                <p className="text-[13px] text-red-400">{error}</p>
              </div>
            )}
            <div>
              <label htmlFor="username" className="block text-[13px] text-gray-300 mb-2.5">Username</label>
              <div className="relative">
                <User className="absolute left-4 top-1/2 -translate-y-1/2 w-[18px] h-[18px] text-gray-500" />
                <input id="username" type="text" value={username} onChange={(e) => setUsername(e.target.value)} placeholder="Enter your username"
                  className="w-full bg-[#0d1220] border border-gray-800 rounded-xl pl-12 pr-4 py-3.5 text-[15px] text-white placeholder:text-gray-600 focus:outline-none focus:border-[#4b73a7] focus:ring-2 focus:ring-[#4b73a7]/20 transition-all" />
              </div>
            </div>
            <div>
              <label htmlFor="password" className="block text-[13px] text-gray-300 mb-2.5">Password</label>
              <div className="relative">
                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-[18px] h-[18px] text-gray-500" />
                <input id="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Enter your password"
                  className="w-full bg-[#0d1220] border border-gray-800 rounded-xl pl-12 pr-4 py-3.5 text-[15px] text-white placeholder:text-gray-600 focus:outline-none focus:border-[#4b73a7] focus:ring-2 focus:ring-[#4b73a7]/20 transition-all" />
              </div>
            </div>
            <button type="submit" disabled={loading}
              className="w-full bg-[#4b73a7] hover:bg-[#3d5f8a] text-white rounded-xl py-3.5 transition-all shadow-lg shadow-[#4b73a7]/25 hover:shadow-[#4b73a7]/40 disabled:opacity-60 flex items-center justify-center gap-2">
              {loading && <Loader2 className="w-5 h-5 animate-spin" />}
              Sign In
            </button>
          </form>

          <div className="mt-8 pt-6 border-t border-gray-800/50 text-center">
            <p className="text-[12px] text-gray-500">Default: admin / admin</p>
          </div>
        </div>
        <p className="text-center mt-6 text-[12px] text-gray-600">Secure radiology workstation</p>
      </motion.div>
    </div>
  );
}
