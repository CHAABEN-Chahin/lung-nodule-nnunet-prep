import { ArrowLeft, Download, Printer, Loader2 } from "lucide-react";
import { motion } from "motion/react";
import { useState, useEffect } from "react";
import { API, getAuthHeaders } from "../App";

interface ReportScreenProps {
  onNavigate: (screen: string) => void;
  caseId: string;
}

interface Nodule {
  nodule_id: string;
  diameter_mm: number;
  volume_mm3: number;
  sphericity: number;
  hu_mean: number;
  hu_std: number;
  hu_min: number;
  hu_max: number;
  hu_median: number;
  hu_p25: number;
  hu_p75: number;
  equiv_radius_mm: number;
  voxel_count: number;
  z_extent_mm: number;
  y_extent_mm: number;
  x_extent_mm: number;
  centroid_z_vox: number;
  original_shape_Elongation?: number;
  original_shape_Flatness?: number;
  original_firstorder_Entropy?: number;
  original_firstorder_Skewness?: number;
  original_glcm_Contrast?: number;
  original_glcm_Correlation?: number;
  original_ngtdm_Busyness?: number;
  original_ngtdm_Complexity?: number;
  [key: string]: unknown;
}

function huDesc(hu: number) {
  if (hu < -750) return "hypodense (composante aerique predominante)";
  if (hu < -300) return "hypodense (densite mixte a predominance aerique)";
  if (hu < 0) return "de densite intermediaire (tissu mou / mixte)";
  if (hu < 100) return "isodense (densite tissulaire)";
  return "hyperdense (composante calcifiee possible)";
}
function sphDesc(s: number) {
  if (s >= 0.85) return "morphologie quasi-spherique reguliere";
  if (s >= 0.70) return "morphologie sub-spherique legerement irreguliere";
  if (s >= 0.50) return "morphologie irreguliere";
  return "morphologie tres irreguliere";
}
function texDesc(n: Nodule) {
  const parts: string[] = [];
  const e = n.original_firstorder_Entropy;
  const c = n.original_glcm_Contrast;
  const h = n.original_glcm_Correlation;
  if (e != null) parts.push(e > 5.5 ? "texture heterogene" : e > 4 ? "texture moderement heterogene" : "texture homogene");
  if (c != null) parts.push(c > 200 ? "contraste interne eleve" : c > 100 ? "contraste interne modere" : "faible contraste interne");
  if (h != null) parts.push(h > 0.2 ? "homogeneite locale satisfaisante" : "heterogeneite locale marquee");
  return parts.join(", ") || "caracteristiques texturales standards";
}
function locDesc(z: number) {
  if (z < 60) return "lobe superieur";
  if (z < 120) return "lobe moyen";
  return "lobe inferieur";
}
function clinicalDesc(n: Nodule) {
  const suf =
    n.diameter_mm >= 8
      ? "La taille de ce nodule necessite une attention clinique particuliere."
      : n.diameter_mm >= 6
        ? "Un suivi rapproche est conseille compte tenu de la taille du nodule."
        : "La taille du nodule est compatible avec une lesion benigne sous surveillance.";
  return `Nodule de ${n.diameter_mm.toFixed(1)} mm de grand axe (volume : ${n.volume_mm3.toFixed(1)} mm3) identifie au niveau du ${locDesc(n.centroid_z_vox)}, presentant une ${sphDesc(n.sphericity)}. La densite scanographique est ${huDesc(n.hu_mean)} (UH moyenne : ${n.hu_mean.toFixed(1)}). L'analyse texturale revele une ${texDesc(n)}. ${suf}`;
}
function recoText(maxD: number): { text: string; border: string; bg: string } {
  if (maxD >= 8)
    return {
      text: "Compte tenu de la taille du nodule le plus volumineux, un avis specialise en pneumologie ou oncologie thoracique est recommande. Un examen complementaire (TEP-scan ou biopsie) pourrait etre envisage.",
      border: "border-[#850808]",
      bg: "bg-[#850808]/5",
    };
  if (maxD >= 6)
    return {
      text: "Un controle TDM thoracique est recommande dans un delai de 6 mois afin de surveiller l'evolution des nodules detectes.",
      border: "border-amber-500",
      bg: "bg-amber-500/5",
    };
  return {
    text: "Les nodules detectes sont de petite taille. Une surveillance annuelle par TDM thoracique est recommandee.",
    border: "border-green-600",
    bg: "bg-green-500/5",
  };
}

function NoduleDetail({ n }: { n: Nodule }) {
  return (
    <div className="mb-8">
      <div className="bg-[#1e3a5f] text-white px-4 py-2.5 mb-3">
        <h4 className="text-[14px]">{n.nodule_id.replace("_", " ").toUpperCase()}</h4>
      </div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <div className="bg-[#d4e3f3] px-3 py-2 mb-2"><h5 className="text-[13px] text-gray-800">Caracteristiques geometriques</h5></div>
          <table className="w-full text-[11px] border-collapse">
            <tbody>
              {([
                ["Grand diametre", `${n.diameter_mm.toFixed(2)} mm`],
                ["Volume", `${n.volume_mm3.toFixed(2)} mm3`],
                ["Rayon equivalent", `${n.equiv_radius_mm.toFixed(3)} mm`],
                ["Sphericite", n.sphericity.toFixed(4)],
                ["Nombre de voxels", String(n.voxel_count)],
                ["Etendue Z / Y / X", `${n.z_extent_mm.toFixed(2)} / ${n.y_extent_mm.toFixed(2)} / ${n.x_extent_mm.toFixed(2)} mm`],
                ...(n.original_shape_Elongation != null ? [["Elongation", n.original_shape_Elongation.toFixed(4)]] : []),
                ...(n.original_shape_Flatness != null ? [["Aplatissement", n.original_shape_Flatness.toFixed(4)]] : []),
              ] as [string, string][]).map(([k, v], i) => (
                <tr key={k} className={`border-b border-gray-200 ${i % 2 === 0 ? "bg-white" : "bg-gray-50"}`}>
                  <td className="py-1.5 px-2 text-gray-700">{k}</td>
                  <td className="py-1.5 px-2 text-right">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <div className="bg-[#d4e3f3] px-3 py-2 mb-2"><h5 className="text-[13px] text-gray-800">Statistiques densitometriques (UH)</h5></div>
          <table className="w-full text-[11px] border-collapse">
            <tbody>
              {([
                ["Moyenne", n.hu_mean.toFixed(2)],
                ["Ecart-type", n.hu_std.toFixed(2)],
                ["Mediane", n.hu_median.toFixed(2)],
                ["Minimum", n.hu_min.toFixed(2)],
                ["Maximum", n.hu_max.toFixed(2)],
                ["Q25 / Q75", `${n.hu_p25.toFixed(1)} / ${n.hu_p75.toFixed(1)}`],
              ] as [string, string][]).map(([k, v], i) => (
                <tr key={k} className={`border-b border-gray-200 ${i % 2 === 0 ? "bg-white" : "bg-gray-50"}`}>
                  <td className="py-1.5 px-2 text-gray-700">{k}</td>
                  <td className="py-1.5 px-2 text-right">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {(n.original_firstorder_Entropy != null || n.original_glcm_Contrast != null) && (
        <>
          <div className="bg-[#d4e3f3] px-3 py-2 mb-2"><h5 className="text-[13px] text-gray-800">Analyse texturale (selection radiomique)</h5></div>
          <table className="w-full text-[11px] border-collapse mb-4">
            <tbody>
              {[
                ["Entropie (1er ordre)", n.original_firstorder_Entropy?.toFixed(4) ?? "-", "Correlation GLCM", n.original_glcm_Correlation?.toFixed(4) ?? "-"],
                ["Contraste GLCM", n.original_glcm_Contrast?.toFixed(4) ?? "-", "Asymetrie (Skewness)", n.original_firstorder_Skewness?.toFixed(4) ?? "-"],
                ["Busyness NGTDM", n.original_ngtdm_Busyness?.toFixed(6) ?? "-", "Complexite NGTDM", n.original_ngtdm_Complexity?.toFixed(4) ?? "-"],
              ].map((row, i) => (
                <tr key={i} className={`border-b border-gray-200 ${i % 2 === 0 ? "bg-white" : "bg-gray-50"}`}>
                  <td className="py-1.5 px-2 text-gray-700 w-1/4">{row[0]}</td>
                  <td className="py-1.5 px-2 text-right w-1/4">{row[1]}</td>
                  <td className="py-1.5 px-2 text-gray-700 w-1/4">{row[2]}</td>
                  <td className="py-1.5 px-2 text-right w-1/4">{row[3]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      <div className="mb-4">
        <h5 className="text-[13px] text-gray-800 mb-2 font-semibold">Description clinique :</h5>
        <p className="text-[12px] text-gray-700 leading-relaxed text-justify">{clinicalDesc(n)}</p>
      </div>
    </div>
  );
}

export function ReportScreen({ onNavigate, caseId }: ReportScreenProps) {
  const [nodules, setNodules] = useState<Nodule[]>([]);
  const [caseInfo, setCaseInfo] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [dlLoading, setDlLoading] = useState(false);
  const today = new Date().toLocaleDateString("fr-FR");

  useEffect(() => {
    Promise.all([
      fetch(`${API}/cases/${caseId}/nodules`, { headers: getAuthHeaders() }).then((r) => r.json()),
      fetch(`${API}/cases/${caseId}`, { headers: getAuthHeaders() }).then((r) => r.json()),
    ])
      .then(([nods, ci]) => {
        setNodules(nods);
        setCaseInfo(ci);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [caseId]);

  const maxD = nodules.length ? Math.max(...nodules.map((n) => n.diameter_mm)) : 0;
  const totalV = nodules.reduce((a, n) => a + n.volume_mm3, 0);
  const reco = recoText(maxD);

  const handleDownload = async () => {
    setDlLoading(true);
    try {
      const res = await fetch(`${API}/cases/${caseId}/report`, { headers: getAuthHeaders() });
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `rapport_${caseId}.pdf`;
      a.click();
    } finally {
      setDlLoading(false);
    }
  };

  const patientName = caseInfo?.patient_name || "";
  const nDossier = caseInfo?.n_dossier || caseId;

  return (
    <div className="min-h-screen bg-[#f8f9fb] flex flex-col">
      <motion.header initial={{ y: -20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ duration: 0.4 }}
        className="h-[72px] bg-white border-b border-gray-200 flex items-center justify-between px-8">
        <div className="flex items-center gap-6">
          <button onClick={() => onNavigate("visualization")} className="flex items-center gap-2 text-gray-600 hover:text-gray-900 transition-colors">
            <ArrowLeft className="w-[18px] h-[18px]" />
            <span className="text-[14px]">Retour</span>
          </button>
          <div className="h-6 w-px bg-gray-300" />
          <div>
            <h1 className="text-[20px] text-gray-900">Compte-Rendu d'Analyse Pulmonaire</h1>
            <p className="text-[13px] text-gray-500">Dossier {nDossier} {patientName && `- ${patientName}`}</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button onClick={() => window.print()} className="flex items-center gap-2 px-4 py-2.5 rounded-lg border border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors">
            <Printer className="w-[16px] h-[16px]" /><span className="text-[14px]">Imprimer</span>
          </button>
          <button onClick={handleDownload} disabled={dlLoading}
            className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-[#4b73a7] hover:bg-[#3d5f8a] text-white transition-colors shadow-sm disabled:opacity-60">
            {dlLoading ? <Loader2 className="w-[16px] h-[16px] animate-spin" /> : <Download className="w-[16px] h-[16px]" />}
            <span className="text-[14px]">Telecharger PDF</span>
          </button>
        </div>
      </motion.header>

      <motion.main initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ duration: 0.5, delay: 0.1 }}
        className="flex-1 p-8 overflow-auto">
        <div className="max-w-[900px] mx-auto bg-white shadow-lg">
          <div className="text-center py-8 border-b-4 border-[#1e3a5f]">
            <h1 className="text-[26px] text-[#1e3a5f] mb-1">HOPITAL HABIB BOURGUIBA SFAX</h1>
            <p className="text-[14px] text-gray-600">Service d'Imagerie Medicale</p>
          </div>
          <div className="px-12 pt-8 pb-6">
            <h2 className="text-[22px] text-[#1e3a5f] text-center pb-3 border-b-2 border-[#1e3a5f] uppercase">
              Compte-Rendu d'Analyse Pulmonaire Automatisee
            </h2>
          </div>
          <div className="px-12 pb-6">
            <table className="w-full text-[13px] border-collapse">
              <tbody>
                {[
                  ["N. Dossier", nDossier, "Date du scanner", caseInfo?.scan_date || today],
                  ["Patient", patientName || "-", "Date du rapport", today],
                  ["Analyse", "Automatisee - Systeme IA Pulmonaire", "", ""],
                ].map((row, i) => (
                  <tr key={i}>
                    <td className="bg-[#e8eef5] px-4 py-2.5 border border-gray-300 w-[25%] font-medium">{row[0]}</td>
                    <td className="px-4 py-2.5 border border-gray-300 w-[25%]">{row[1]}</td>
                    {row[2] && <td className="bg-[#e8eef5] px-4 py-2.5 border border-gray-300 w-[25%] font-medium">{row[2]}</td>}
                    {row[3] && <td className="px-4 py-2.5 border border-gray-300 w-[25%]">{row[3]}</td>}
                    {!row[2] && <td colSpan={2} className="border border-gray-300"></td>}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="px-12 pb-6">
            <h3 className="text-[16px] text-[#1e3a5f] mb-3 pb-1 border-b border-[#1e3a5f]">Technique</h3>
            <p className="text-[12px] text-gray-700 leading-relaxed italic">
              Analyse automatisee realisee a partir du volume TDM thoracique reconstruit en 3D. Detection et segmentation
              des nodules pulmonaires par reseau de neurones convolutif (nnU-Net v2). Extraction des caracteristiques
              morphologiques, densitometriques et texturales par analyse radiomique (PyRadiomics).
            </p>
          </div>
          <div className="px-12 pb-6">
            <h3 className="text-[16px] text-[#1e3a5f] mb-3 pb-1 border-b border-[#1e3a5f]">Resultats - Vue d'ensemble</h3>
            {loading ? (
              <div className="flex items-center gap-2 text-gray-500 py-4"><Loader2 className="w-4 h-4 animate-spin" /><span>Chargement...</span></div>
            ) : (
              <>
                <p className="text-[13px] text-gray-800 mb-4">{nodules.length} nodule(s) pulmonaire(s) detecte(s) et analyse(s).</p>
                <table className="w-full text-[12px] border-collapse">
                  <thead>
                    <tr className="bg-[#1e3a5f] text-white">
                      {["Nodule", "Diametre (mm)", "Volume (mm3)", "Sphericite", "UH Moyenne", "Localisation (z)"].map((h) => (
                        <th key={h} className="px-3 py-2.5 text-left border border-gray-400">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {nodules.map((n, i) => (
                      <tr key={n.nodule_id} className={i % 2 === 0 ? "bg-[#e8eef5]" : "bg-white"}>
                        <td className="px-3 py-2 border border-gray-300">{n.nodule_id.replace("_", " ").toUpperCase()}</td>
                        <td className="px-3 py-2 border border-gray-300">{n.diameter_mm.toFixed(1)}</td>
                        <td className="px-3 py-2 border border-gray-300">{n.volume_mm3.toFixed(1)}</td>
                        <td className="px-3 py-2 border border-gray-300">{n.sphericity.toFixed(3)}</td>
                        <td className="px-3 py-2 border border-gray-300">{n.hu_mean.toFixed(1)}</td>
                        <td className="px-3 py-2 border border-gray-300">{n.centroid_z_vox.toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}
          </div>
          {!loading && nodules.length > 0 && (
            <div className="px-12 pb-6">
              <h3 className="text-[16px] text-[#1e3a5f] mb-4 pb-1 border-b border-[#1e3a5f]">Resultats - Analyse detaillee par nodule</h3>
              {nodules.map((n) => <NoduleDetail key={n.nodule_id} n={n} />)}
            </div>
          )}
          {!loading && nodules.length > 0 && (
            <div className="px-12 pb-6">
              <h3 className="text-[16px] text-[#1e3a5f] mb-3 pb-1 border-b border-[#1e3a5f]">Conclusion</h3>
              <p className="text-[13px] text-gray-800 mb-3">
                L'analyse automatisee du volume TDM thoracique a permis d'identifier{" "}
                <strong>{nodules.length} nodule(s) pulmonaire(s)</strong> de taille significative (diametre &gt; 3 mm).
                Le grand diametre maximal observe est de <strong>{maxD.toFixed(1)} mm</strong> et le volume nodulaire total
                est de <strong>{totalV.toFixed(1)} mm3</strong>.
              </p>
              {nodules.map((n) => (
                <p key={n.nodule_id} className="text-[13px] text-gray-800 mb-1">
                  - <strong>{n.nodule_id.replace("_", " ").toUpperCase()}</strong> : {n.diameter_mm.toFixed(1)} mm,
                  sphericite {n.sphericity.toFixed(3)}, densite moyenne {n.hu_mean.toFixed(1)} UH.
                </p>
              ))}
              <div className={`border-2 ${reco.border} ${reco.bg} p-4 mt-4 mb-4`}>
                <p className="text-[13px] text-gray-900"><strong>Recommandation :</strong> {reco.text}</p>
              </div>
              <p className="text-[11px] text-gray-600 italic mb-6">
                Ce compte-rendu est genere automatiquement par le systeme d'analyse pulmonaire IA.
                Il doit etre valide et signe par un medecin radiologue avant toute utilisation clinique.
              </p>
            </div>
          )}
          <div className="px-12 pb-8 pt-4 border-t border-gray-300">
            <div className="flex justify-between items-end text-[11px] text-gray-700">
              <div>Valide le {today}</div>
              <div className="text-right">
                <p className="font-semibold">Dr. Radiologist</p>
                <p>Hopital Habib Bourguiba Sfax</p>
              </div>
            </div>
          </div>
        </div>
      </motion.main>
    </div>
  );
}
