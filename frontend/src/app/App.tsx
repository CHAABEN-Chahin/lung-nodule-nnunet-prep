import { useState, useEffect } from "react";
import { LoginScreen } from "./components/LoginScreen";
import { Dashboard } from "./components/Dashboard";
import { VisualizationScreen } from "./components/VisualizationScreen";
import { ReportScreen } from "./components/ReportScreen";
import { SettingsScreen } from "./components/SettingsScreen";

type Screen = "login" | "dashboard" | "visualization" | "report" | "settings";

export interface UserInfo {
  id: number;
  username: string;
  full_name: string;
  department: string;
  email: string;
  hospital_name: string;
  service: string;
}

export const API = "http://localhost:5000/api";

export function getAuthHeaders(): Record<string, string> {
  const token = localStorage.getItem("nodulis_token");
  return token ? { Authorization: `Bearer ${token}`, "Content-Type": "application/json" } : { "Content-Type": "application/json" };
}

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<Screen>("login");
  const [selectedCaseId, setSelectedCaseId] = useState<string>("");
  const [user, setUser] = useState<UserInfo | null>(null);

  useEffect(() => {
    const token = localStorage.getItem("nodulis_token");
    if (token) {
      fetch(`${API}/auth/me`, { headers: getAuthHeaders() })
        .then((r) => (r.ok ? r.json() : Promise.reject()))
        .then((u: UserInfo) => {
          setUser(u);
          setCurrentScreen("dashboard");
        })
        .catch(() => {
          localStorage.removeItem("nodulis_token");
        });
    }
  }, []);

  const handleLogin = (token: string, userInfo: UserInfo) => {
    localStorage.setItem("nodulis_token", token);
    setUser(userInfo);
    setCurrentScreen("dashboard");
  };

  const handleLogout = () => {
    fetch(`${API}/auth/logout`, { method: "POST", headers: getAuthHeaders() }).catch(() => {});
    localStorage.removeItem("nodulis_token");
    setUser(null);
    setCurrentScreen("login");
  };

  const handleNavigate = (screen: string, caseId?: string) => {
    if (caseId) setSelectedCaseId(caseId);
    setCurrentScreen(screen as Screen);
  };

  return (
    <div className="size-full">
      {currentScreen === "login" && <LoginScreen onLogin={handleLogin} />}
      {currentScreen === "dashboard" && <Dashboard onNavigate={handleNavigate} onLogout={handleLogout} user={user} />}
      {currentScreen === "visualization" && <VisualizationScreen onNavigate={handleNavigate} caseId={selectedCaseId} />}
      {currentScreen === "report" && <ReportScreen onNavigate={handleNavigate} caseId={selectedCaseId} />}
      {currentScreen === "settings" && <SettingsScreen onNavigate={handleNavigate} user={user} onUserUpdate={setUser} />}
    </div>
  );
}
