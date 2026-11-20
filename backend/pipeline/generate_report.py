import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY

DARK_BLUE = colors.HexColor("#1B3A5C")
MID_BLUE = colors.HexColor("#2E6DA4")
LIGHT_BLUE = colors.HexColor("#EAF2FB")
LIGHT_GRAY = colors.HexColor("#F5F5F5")
MED_GRAY = colors.HexColor("#CCCCCC")
DARK_GRAY = colors.HexColor("#444444")
WHITE = colors.white
RED_FLAG = colors.HexColor("#C0392B")
ORANGE_FLAG = colors.HexColor("#E67E22")
GREEN_FLAG = colors.HexColor("#27AE60")


def make_styles():
    return {
        "hospital": ParagraphStyle("hospital", fontName="Helvetica-Bold", fontSize=13, textColor=DARK_BLUE, alignment=TA_CENTER, spaceAfter=2),
        "service": ParagraphStyle("service", fontName="Helvetica", fontSize=10, textColor=DARK_BLUE, alignment=TA_CENTER, spaceAfter=2),
        "report_title": ParagraphStyle("report_title", fontName="Helvetica-Bold", fontSize=13, textColor=DARK_BLUE, alignment=TA_CENTER, spaceBefore=8, spaceAfter=8, underline=True),
        "section_title": ParagraphStyle("section_title", fontName="Helvetica-Bold", fontSize=11, textColor=DARK_BLUE, spaceBefore=10, spaceAfter=4),
        "nodule_title": ParagraphStyle("nodule_title", fontName="Helvetica-Bold", fontSize=10, textColor=WHITE, spaceBefore=0, spaceAfter=0, leftIndent=6),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, textColor=DARK_GRAY, spaceBefore=3, spaceAfter=3, alignment=TA_JUSTIFY, leading=14),
        "body_bold": ParagraphStyle("body_bold", fontName="Helvetica-Bold", fontSize=9.5, textColor=DARK_GRAY, spaceBefore=3, spaceAfter=3),
        "italic": ParagraphStyle("italic", fontName="Helvetica-Oblique", fontSize=9.5, textColor=DARK_GRAY, spaceBefore=3, spaceAfter=3, alignment=TA_JUSTIFY, leading=14),
        "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8.5, textColor=DARK_GRAY, spaceBefore=2, spaceAfter=2),
        "conclusion": ParagraphStyle("conclusion", fontName="Helvetica", fontSize=9.5, textColor=DARK_GRAY, spaceBefore=4, spaceAfter=4, alignment=TA_JUSTIFY, leading=14),
    }


def hu_description(hu_mean):
    if hu_mean < -750: return "hypodense (composante aerique predominante)"
    if hu_mean < -300: return "hypodense (densite mixte a predominance aerique)"
    if hu_mean < 0:    return "de densite intermediaire (tissu mou / mixte)"
    if hu_mean < 100:  return "isodense (densite tissulaire)"
    return "hyperdense (composante calcifiee possible)"


def sphericity_description(sph):
    if sph >= 0.85: return "morphologie quasi-spherique reguliere"
    if sph >= 0.70: return "morphologie sub-spherique legerement irreguliere"
    if sph >= 0.50: return "morphologie irreguliere"
    return "morphologie tres irreguliere"


def texture_description(nodule):
    entropy = nodule.get("original_firstorder_Entropy")
    contrast = nodule.get("original_glcm_Contrast")
    homogeneity = nodule.get("original_glcm_Idm")
    parts = []
    if entropy is not None:
        parts.append("texture heterogene" if entropy > 5.5 else "texture moderement heterogene" if entropy > 4.0 else "texture homogene")
    if contrast is not None:
        parts.append("contraste interne eleve" if contrast > 200 else "contraste interne modere" if contrast > 100 else "faible contraste interne")
    if homogeneity is not None:
        parts.append("homogeneite locale satisfaisante" if homogeneity > 0.2 else "heterogeneite locale marquee")
    return ", ".join(parts) if parts else "caracteristiques texturales standards"


def generate_description(nodule):
    diam = nodule.get("diameter_mm", 0)
    vol = nodule.get("volume_mm3", 0)
    sph = nodule.get("sphericity", nodule.get("original_shape_Sphericity", 0))
    hu_mean = nodule.get("hu_mean", 0)
    cz = nodule.get("centroid_z_vox", 0)

    location = "lobe superieur" if cz < 60 else "lobe moyen" if cz < 120 else "lobe inferieur"

    desc = (
        f"Nodule de {diam:.1f} mm de grand axe (volume : {vol:.1f} mm3) "
        f"identifie au niveau du {location}, presentant une {sphericity_description(sph)}. "
        f"La densite scanographique est {hu_description(hu_mean)} "
        f"(UH moyenne : {hu_mean:.1f}). "
        f"L'analyse texturale revele une {texture_description(nodule)}. "
    )

    if diam >= 8:
        desc += "La taille de ce nodule necessite une attention clinique particuliere."
    elif diam >= 6:
        desc += "Un suivi rapproche est conseille compte tenu de la taille du nodule."
    else:
        desc += "La taille du nodule est compatible avec une lesion benigne sous surveillance."

    return desc


def get_recommendation(max_diam):
    if max_diam >= 8:
        return (
            "Compte tenu de la taille du nodule le plus volumineux, "
            "un avis specialise en pneumologie ou oncologie thoracique est recommande. "
            "Un examen complementaire (TEP-scan ou biopsie) pourrait etre envisage.",
            RED_FLAG,
        )
    if max_diam >= 6:
        return (
            "Un controle TDM thoracique est recommande dans un delai de 6 mois "
            "afin de surveiller l'evolution des nodules detectes.",
            ORANGE_FLAG,
        )
    return (
        "Les nodules detectes sont de petite taille. "
        "Une surveillance annuelle par TDM thoracique est recommandee.",
        GREEN_FLAG,
    )


def build_header(styles, info):
    elements = []
    elements.append(Paragraph(info["hospital_name"].upper(), styles["hospital"]))
    elements.append(Paragraph(info["service"], styles["service"]))
    elements.append(Paragraph(info["doctor_name"], styles["service"]))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(HRFlowable(width="100%", thickness=2, color=DARK_BLUE))
    elements.append(Spacer(1, 0.2 * cm))
    elements.append(Paragraph("COMPTE-RENDU D'ANALYSE PULMONAIRE AUTOMATISEE", styles["report_title"]))
    elements.append(HRFlowable(width="100%", thickness=1, color=MID_BLUE))
    elements.append(Spacer(1, 0.3 * cm))
    return elements


def build_patient_table(styles, info):
    today = datetime.now().strftime("%d/%m/%Y")
    data = [
        [Paragraph("<b>N Dossier</b>", styles["small"]), Paragraph(info["n_dossier"], styles["small"]),
         Paragraph("<b>Date du scanner</b>", styles["small"]), Paragraph(info["scan_date"], styles["small"])],
        [Paragraph("<b>ID Patient</b>", styles["small"]), Paragraph(info.get("patient_name", ""), styles["small"]),
         Paragraph("<b>Date du rapport</b>", styles["small"]), Paragraph(today, styles["small"])],
        [Paragraph("<b>Medecin</b>", styles["small"]), Paragraph(info["doctor_name"], styles["small"]),
         Paragraph("<b>Analyse</b>", styles["small"]), Paragraph("Automatisee - Systeme IA Pulmonaire", styles["small"])],
    ]
    col_widths = [3.2 * cm, 5.5 * cm, 3.8 * cm, 5.5 * cm]
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE), ("BACKGROUND", (2, 0), (2, -1), LIGHT_BLUE),
        ("BACKGROUND", (1, 0), (1, -1), WHITE), ("BACKGROUND", (3, 0), (3, -1), WHITE),
        ("BOX", (0, 0), (-1, -1), 0.5, MED_GRAY), ("INNERGRID", (0, 0), (-1, -1), 0.5, MED_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return [t, Spacer(1, 0.4 * cm)]


def build_technique(styles):
    elements = []
    elements.append(Paragraph("Technique", styles["section_title"]))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=MID_BLUE))
    elements.append(Spacer(1, 0.2 * cm))
    elements.append(Paragraph(
        "Analyse automatisee realisee a partir du volume TDM thoracique reconstruit en 3D. "
        "Detection et segmentation des nodules pulmonaires par reseau de neurones convolutif (nnU-Net v2). "
        "Extraction des caracteristiques morphologiques, densitometriques et texturales par analyse "
        "radiomique (PyRadiomics). Les resultats presentes sont issus du traitement automatique "
        "et doivent etre interpretes par un medecin radiologue qualifie.",
        styles["italic"]
    ))
    elements.append(Spacer(1, 0.3 * cm))
    return elements


def build_summary_table(styles, nodules):
    elements = []
    elements.append(Paragraph("Resultats - Vue d'ensemble", styles["section_title"]))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=MID_BLUE))
    elements.append(Spacer(1, 0.2 * cm))
    elements.append(Paragraph(f"{len(nodules)} nodule(s) pulmonaire(s) detecte(s) et analyse(s).", styles["body"]))
    elements.append(Spacer(1, 0.2 * cm))

    header = [
        Paragraph("<b>Nodule</b>", styles["small"]), Paragraph("<b>Diametre (mm)</b>", styles["small"]),
        Paragraph("<b>Volume (mm3)</b>", styles["small"]), Paragraph("<b>Sphericite</b>", styles["small"]),
        Paragraph("<b>UH Moyenne</b>", styles["small"]), Paragraph("<b>Localisation (Z vox)</b>", styles["small"]),
    ]
    rows = [header]
    for n in nodules:
        nid = n.get("nodule_id", "?").replace("_", " ").capitalize()
        rows.append([
            Paragraph(nid, styles["small"]),
            Paragraph(f"{n.get('diameter_mm', 0):.1f}", styles["small"]),
            Paragraph(f"{n.get('volume_mm3', 0):.1f}", styles["small"]),
            Paragraph(f"{n.get('sphericity', 0):.3f}", styles["small"]),
            Paragraph(f"{n.get('hu_mean', 0):.1f}", styles["small"]),
            Paragraph(f"{n.get('centroid_z_vox', 0):.1f}", styles["small"]),
        ])

    col_widths = [3.0 * cm, 3.2 * cm, 3.2 * cm, 3.0 * cm, 3.0 * cm, 3.6 * cm]
    t = Table(rows, colWidths=col_widths)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE), ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BOX", (0, 0), (-1, -1), 0.5, MED_GRAY), ("INNERGRID", (0, 0), (-1, -1), 0.5, MED_GRAY),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]
    for i in range(1, len(rows)):
        bg = LIGHT_BLUE if i % 2 == 0 else WHITE
        style.append(("BACKGROUND", (0, i), (-1, i), bg))
    t.setStyle(TableStyle(style))
    elements.append(t)
    elements.append(Spacer(1, 0.4 * cm))
    return elements


def build_nodule_detail(styles, nodule, index):
    elements = []
    nid = nodule.get("nodule_id", f"nodule_{index:03d}").replace("_", " ").capitalize()
    diam = nodule.get("diameter_mm", 0)
    vol = nodule.get("volume_mm3", 0)
    sph = nodule.get("sphericity", nodule.get("original_shape_Sphericity", 0))
    hu_m = nodule.get("hu_mean", 0)
    hu_s = nodule.get("hu_std", 0)
    hu_mn = nodule.get("hu_min", 0)
    hu_mx = nodule.get("hu_max", 0)
    hu_md = nodule.get("hu_median", 0)
    vox = nodule.get("voxel_count", 0)
    r_eq = nodule.get("equiv_radius_mm", 0)
    ze = nodule.get("z_extent_mm", 0)
    ye = nodule.get("y_extent_mm", 0)
    xe = nodule.get("x_extent_mm", 0)
    elong = nodule.get("original_shape_Elongation")
    flat = nodule.get("original_shape_Flatness")
    entropy = nodule.get("original_firstorder_Entropy")
    skew = nodule.get("original_firstorder_Skewness")
    glcm_contrast = nodule.get("original_glcm_Contrast")
    glcm_corr = nodule.get("original_glcm_Correlation")

    title_data = [[Paragraph(f"<b>{nid}</b>", styles["nodule_title"])]]
    title_table = Table(title_data, colWidths=[18 * cm])
    title_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK_BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(title_table)
    elements.append(Spacer(1, 0.2 * cm))

    geo_data = [
        [Paragraph("<b>Caracteristiques geometriques</b>", styles["small"]), ""],
        ["Grand diametre", f"{diam:.2f} mm"],
        ["Volume", f"{vol:.2f} mm3"],
        ["Rayon equivalent", f"{r_eq:.3f} mm"],
        ["Sphericite", f"{sph:.4f}"],
        ["Nombre de voxels", f"{vox}"],
        ["Etendue Z / Y / X", f"{ze:.2f} / {ye:.2f} / {xe:.2f} mm"],
    ]
    if elong is not None:
        geo_data.append(["Elongation", f"{elong:.4f}"])
    if flat is not None:
        geo_data.append(["Aplatissement", f"{flat:.4f}"])

    hu_data = [
        [Paragraph("<b>Statistiques densitometriques (UH)</b>", styles["small"]), ""],
        ["Moyenne", f"{hu_m:.2f}"],
        ["Ecart-type", f"{hu_s:.2f}"],
        ["Mediane", f"{hu_md:.2f}"],
        ["Minimum", f"{hu_mn:.2f}"],
        ["Maximum", f"{hu_mx:.2f}"],
        ["Q25 / Q75", f"{nodule.get('hu_p25', 0):.1f} / {nodule.get('hu_p75', 0):.1f}"],
    ]

    def small_table(data):
        t = Table(data, colWidths=[4.2 * cm, 3.8 * cm])
        style = [
            ("SPAN", (0, 0), (1, 0)), ("BACKGROUND", (0, 0), (1, 0), LIGHT_BLUE),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("BOX", (0, 0), (-1, -1), 0.5, MED_GRAY), ("INNERGRID", (0, 0), (-1, -1), 0.5, MED_GRAY),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("ALIGN", (1, 1), (1, -1), "RIGHT"),
        ]
        for i in range(1, len(data)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), LIGHT_GRAY))
        t.setStyle(TableStyle(style))
        return t

    two_col = Table(
        [[small_table(geo_data), small_table(hu_data)]],
        colWidths=[8.4 * cm, 8.4 * cm], hAlign="LEFT",
    )
    two_col.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(two_col)
    elements.append(Spacer(1, 0.2 * cm))

    if entropy is not None or glcm_contrast is not None:
        tex_rows = [[Paragraph("<b>Analyse texturale (selection radiomique)</b>", styles["small"]), "", "", ""]]
        if entropy is not None:
            tex_rows.append(["Entropie (1er ordre)", f"{entropy:.4f}",
                             "Correlation GLCM", f"{glcm_corr:.4f}" if glcm_corr else "N/A"])
        if glcm_contrast is not None:
            tex_rows.append(["Contraste GLCM", f"{glcm_contrast:.4f}",
                             "Asymetrie (Skewness)", f"{skew:.4f}" if skew else "N/A"])
        ngtdm_b = nodule.get("original_ngtdm_Busyness")
        ngtdm_c = nodule.get("original_ngtdm_Complexity")
        if ngtdm_b is not None:
            tex_rows.append(["Busyness NGTDM", f"{ngtdm_b:.6f}",
                             "Complexite NGTDM", f"{ngtdm_c:.4f}" if ngtdm_c else "N/A"])

        tex_table = Table(tex_rows, colWidths=[4.5 * cm, 3.5 * cm, 4.5 * cm, 3.5 * cm])
        tex_table.setStyle(TableStyle([
            ("SPAN", (0, 0), (3, 0)), ("BACKGROUND", (0, 0), (3, 0), LIGHT_BLUE),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("BOX", (0, 0), (-1, -1), 0.5, MED_GRAY), ("INNERGRID", (0, 0), (-1, -1), 0.5, MED_GRAY),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ]))
        elements.append(tex_table)
        elements.append(Spacer(1, 0.2 * cm))

    elements.append(Paragraph("<b>Description clinique :</b>", styles["body_bold"]))
    elements.append(Paragraph(generate_description(nodule), styles["body"]))
    elements.append(Spacer(1, 0.35 * cm))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=MED_GRAY, dash=(3, 3)))
    elements.append(Spacer(1, 0.35 * cm))
    return elements


def build_conclusion(styles, nodules):
    elements = []
    elements.append(Paragraph("Conclusion", styles["section_title"]))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=MID_BLUE))
    elements.append(Spacer(1, 0.2 * cm))

    n = len(nodules)
    max_diam = max((nd.get("diameter_mm", 0) for nd in nodules), default=0)
    total_vol = sum(nd.get("volume_mm3", 0) for nd in nodules)

    elements.append(Paragraph(
        f"L'analyse automatisee du volume TDM thoracique a permis d'identifier "
        f"<b>{n} nodule(s) pulmonaire(s)</b> de taille significative (diametre &gt; 3 mm). "
        f"Le grand diametre maximal observe est de <b>{max_diam:.1f} mm</b> et le volume "
        f"nodulaire total est de <b>{total_vol:.1f} mm3</b>.",
        styles["conclusion"]
    ))
    elements.append(Spacer(1, 0.15 * cm))

    for nd in nodules:
        nid = nd.get("nodule_id", "?").replace("_", " ").capitalize()
        elements.append(Paragraph(
            f"- <b>{nid}</b> : {nd.get('diameter_mm', 0):.1f} mm, "
            f"sphericite {nd.get('sphericity', 0):.3f}, "
            f"densite moyenne {nd.get('hu_mean', 0):.1f} UH.",
            styles["conclusion"]
        ))

    elements.append(Spacer(1, 0.3 * cm))

    reco_text, reco_color = get_recommendation(max_diam)
    reco_data = [[Paragraph(f"<b>Recommandation :</b> {reco_text}", styles["conclusion"])]]
    reco_table = Table(reco_data, colWidths=[18 * cm])
    reco_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EAF2FB")),
        ("BOX", (0, 0), (-1, -1), 1.5, reco_color),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(reco_table)
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph(
        "Ce compte-rendu est genere automatiquement par le systeme d'analyse pulmonaire IA. "
        "Il doit etre valide et signe par un medecin radiologue avant toute utilisation clinique.",
        styles["italic"]
    ))
    return elements


def build_signature(styles, info):
    elements = []
    elements.append(Spacer(1, 0.8 * cm))
    elements.append(HRFlowable(width="100%", thickness=1, color=DARK_BLUE))
    elements.append(Spacer(1, 0.3 * cm))
    today = datetime.now().strftime("%d/%m/%Y a %H:%M")

    sig_data = [[
        Paragraph(f"Valide le {today}", styles["small"]),
        Paragraph(
            f"<b>{info['doctor_name']}</b><br/>{info['hospital_name']}",
            ParagraphStyle("sig_right", fontName="Helvetica-Bold", fontSize=9, alignment=TA_RIGHT),
        ),
    ]]
    sig_table = Table(sig_data, colWidths=[9 * cm, 9 * cm])
    sig_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    elements.append(sig_table)
    return elements


def generate_report(summary_path, output_path, patient_info):
    with open(summary_path, "r") as f:
        nodules = json.load(f)

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Compte-Rendu Analyse Pulmonaire",
        author=patient_info.get("doctor_name", ""),
    )

    styles = make_styles()
    story = []

    story += build_header(styles, patient_info)
    story.append(Spacer(1, 0.3 * cm))
    story += build_patient_table(styles, patient_info)
    story += build_technique(styles)
    story += build_summary_table(styles, nodules)

    story.append(Paragraph("Resultats - Analyse detaillee par nodule", styles["section_title"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_BLUE))
    story.append(Spacer(1, 0.3 * cm))

    for i, nodule in enumerate(nodules, start=1):
        story += build_nodule_detail(styles, nodule, i)

    story += build_conclusion(styles, nodules)
    story += build_signature(styles, patient_info)

    doc.build(story)
    print(f"  Report generated: {output_path}")
