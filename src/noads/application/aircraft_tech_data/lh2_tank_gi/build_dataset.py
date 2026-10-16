"""Build the LH2 tank gravimetric index (GI) dataset for NOADS parameterization.

Every row is one reported GI value (or range). Columns:
 id, source_key, year_published, data_type, gi_scope, gi, gi_low, gi_high,
 m_h2_kg_per_tank, m_h2_basis, n_tanks, m_h2_kg_aircraft, diameter_m, volume_m3,
 insulation, wall_material, tech_year, aircraft_class, conditions, notes,
 confidence, url
data_type: hardware_measured | hardware_claim | design_study | projection | target | requirement
gi_scope: tank (m_H2/(m_H2+m_tank)) | system (adds fuel system: pipes, pumps, conditioning) | unclear
m_h2_basis: stated | derived (computed from stated numbers) | estimated (my estimate, see notes)
tech_year: year the value applies to (hardware test year, EIS/TRL6 year, or study tech basis)
"""
import csv

U = {
 "flyzero_cryo": "https://www.ati.org.uk/wp-content/uploads/2022/03/FZO-PPN-COM-0027-Cryogenic-Hydrogen-Fuel-System-and-Storage-Roadmap-Report.pdf",
 "flyzero_concepts": "https://www.ati.org.uk/wp-content/uploads/2022/03/FZO-AIN-REP-0007-FlyZero-Zero-Carbon-Emission-Aircraft-Concepts.pdf",
 "flyzero_energy": "https://www.ati.org.uk/wp-content/uploads/2021/09/FZ_0_6.1-Primary-Energy-Source-Comparison-and-Selection-FINAL-230921.pdf",
 "huete2021": "https://dspace.lib.cranfield.ac.uk/server/api/core/bitstreams/e8e2178c-12b3-4295-aeff-a39e32ebeb8e/content",
 "huete2022": "https://www.cambridge.org/core/journals/aeronautical-journal/article/impact-of-tank-gravimetric-efficiency-on-propulsion-system-integration-for-a-firstgeneration-hydrogen-civil-airliner/CC45233F25B8A29C5916BFDD552D2056",
 "verstraete2010": "https://www.academia.edu/59226845/Hydrogen_fuel_tanks_for_subsonic_transport_aircraft",
 "jagtap2023": "https://engrxiv.org/preprint/download/4285/7844/6473",
 "mckinsey2020": "https://www.clean-hydrogen.europa.eu/system/files/2020-06/20200507_Hydrogen%2520Powered%2520Aviation%2520report_FINAL%2520web%2520%2528ID%25208706035%2529.pdf",
 "ca_workshop2021": "https://www.clean-aviation.eu/sites/default/files/2021-07/CA-CH%20workshop%20-%20key%20messages.pdf",
 "iata2023": "https://www.iata.org/contentassets/8d19e716636a47c184e7221c77563c93/aircraft-technology-net-zero-roadmap.pdf",
 "icct2022": "https://theicct.org/wp-content/uploads/2022/01/LH2-aircraft-white-paper-A4-v4.pdf",
 "mit2023": "https://arxiv.org/pdf/2309.14629",
 "adler2023": "https://www.sciencedirect.com/science/article/pii/S0376042123000386",
 "tiwari2024": "https://www.sciencedirect.com/science/article/pii/S0360319923065631",
 "saias2022": "https://www.sciencedirect.com/science/article/pii/S0360319922030099",
 "dlr2026": "https://doi.org/10.3390/hydrogen7030120",
 "licheva2023": "https://www.icas.org/icas_archive/icas2024/data/papers/icas2024_0193_paper.pdf",
 "mazzoni2024": "https://iris.polito.it/retrieve/handle/11583/2994563/762669",
 "svensson2023": "https://www.icas.org/icas_archive/icas2024/data/papers/icas2024_1078_paper.pdf",
 "winnefeld2018": "https://publikationen.bibliothek.kit.edu/1000159831/151025714",
 "izea2025": "https://www.sciencedirect.com/science/article/pii/S0306261925007846",
 "soton2025": "https://eprints.soton.ac.uk/509290/1/1-s2.0-S0360319925060458-main.pdf",
 "aerospace2024": "https://www2.mdpi.com/2226-4310/11/2/161",
 "parello2024": "https://www.icas.org/icas_archive/icas2024/data/papers/icas2024_0439_paper.pdf",
 "onera2024": "https://www.icas.org/icas_archive/icas2024/data/papers/icas2024_0826_paper.pdf",
 "chalmers2022": "https://www.icas.org/icas_archive/ICAS2022/data/papers/ICAS2022_0496_paper.pdf",
 "gtl2024": "https://www.businesswire.com/news/home/20240313965264/en/GTL-Announces-Pioneering-Validation-of-LH2-Composite-Dewar-Tanks-for-Aviation-Applications",
 "cocolih2t2026": "https://doi.org/10.3390/engproc2026133165",
 "overleaf": "https://cordis.europa.eu/project/id/101056818",
 "ca_call_fta07": "https://www.fundingprogrammesportal.gov.cy/en/call/innovative-light-weight-and-reliable-liquid-hydrogen-tank-en-2026/",
 "ca_call_hpa02": "https://www.horizon-europe.gouv.fr/components-development-and-experimental-testing-onboard-liquid-hydrogen-supply-and-conditioning",
 "uav2025": "https://www.sciencedirect.com/science/article/pii/S0360319925049638",
 "nasa_hale2009": "https://ntrs.nasa.gov/api/citations/20090013674/downloads/20090013674.pdf",
}

rows = []
def add(**k):
    k.setdefault("gi_low", ""); k.setdefault("gi_high", "")
    for f in ["m_h2_kg_per_tank","m_h2_basis","n_tanks","m_h2_kg_aircraft","diameter_m","volume_m3",
              "insulation","wall_material","aircraft_class","conditions","notes"]:
        k.setdefault(f, "")
    k["url"] = U[k["source_key"]]
    rows.append(k)

# ---------------- FlyZero cryogenic roadmap (ATI 2022) ----------------
# Aircraft design-mission LH2 from FlyZero concepts report: regional 1158 kg, narrowbody 3903 kg, midsize 16743 kg.
fz = [
 # (concept, tank label, insulation, m_aircraft, n_tanks, m_per_tank, basis_note, values 2026/2030/2050)
 ("regional","aft tank 1","vacuum/MLI",1158,2,579,"equal split of design-mission LH2 over 2 aft tanks (estimated)",(0.57,0.75,0.75)),
 ("regional","aft tank 2","vacuum/MLI",1158,2,579,"equal split of design-mission LH2 over 2 aft tanks (estimated)",(0.54,0.72,0.72)),
 ("narrowbody","aft tank 1","foam",3903,2,1952,"equal split of design-mission LH2 over 2 aft tanks (estimated)",(0.67,0.76,0.77)),
 ("narrowbody","aft tank 2","foam",3903,2,1952,"equal split of design-mission LH2 over 2 aft tanks (estimated)",(0.63,0.72,0.73)),
 ("midsize","aft tank","foam",16743,3,11160,"assumed aft tank holds ~2/3 of LH2 (estimated; FlyZero gives no split)",(0.82,0.89,0.90)),
 ("midsize","delta tank","foam",16743,3,2790,"assumed 2 forward delta tanks share ~1/3 of LH2 (estimated); boxed geometry penalised",(0.45,0.65,0.66)),
]
mat = {2026:"aluminium", 2030:"composite", 2050:"composite (improved foam k and density)"}
for concept, lab, ins, mac, nt, mpt, bn, vals in fz:
    for yr, v in zip((2026,2030,2050), vals):
        add(id=f"FZ-{concept}-{lab.replace(' ','')}-{yr}", source_key="flyzero_cryo", year_published=2022,
            data_type="projection", gi_scope="tank", gi=v, m_h2_kg_per_tank=mpt, m_h2_basis="estimated",
            n_tanks=nt, m_h2_kg_aircraft=mac, insulation=ins, wall_material=mat[yr], tech_year=yr,
            aircraft_class=concept, conditions="TRL6 year 2026; FlyZero tank-only definition (excl. pumps, sensors, pipework)",
            notes=f"FlyZero {concept} {lab}. {bn}. m_h2_kg_aircraft is design-mission fuel (tank capacity is larger).",
            confidence="high (GI) / low (per-tank mass)")
for concept, mac, vals in [("regional",1158,(0.47,0.61,0.64)),("narrowbody",3903,(0.58,0.66,0.69)),("midsize",16743,(0.58,0.72,0.75))]:
    for yr, v in zip((2026,2030,2050), vals):
        add(id=f"FZ-{concept}-total-{yr}", source_key="flyzero_cryo", year_published=2022, data_type="projection",
            gi_scope="system", gi=v, m_h2_kg_aircraft=mac, m_h2_basis="stated", insulation="vacuum/MLI" if concept=="regional" else "foam",
            wall_material=mat[yr], tech_year=yr, aircraft_class=concept,
            conditions="FlyZero total GI = fuel/(fuel + fuel system + empty tanks)",
            notes="Aircraft-level total incl. fuel system equipment, pipework, active cooling if fitted. Pair with tank rows to back out fuel-system mass fraction.",
            confidence="high")
add(id="FZ-energy-2021", source_key="flyzero_energy", year_published=2021, data_type="design_study", gi_scope="tank", gi=0.60,
    tech_year=2030, aircraft_class="generic", conditions="1.5 bar, 20 K",
    notes="'Conservative assumption based on the sizing methods used in Brewer'. Early FlyZero screening value.", confidence="medium")

# ---------------- Huete & Pilidis 2021 (Cranfield, IJHE) ----------------
add(id="HU21-100m3-foam", source_key="huete2021", year_published=2021, data_type="design_study", gi_scope="tank", gi=0.660,
    m_h2_kg_per_tank=6128, m_h2_basis="stated", n_tanks=1, diameter_m=4.0, volume_m3=100, insulation="foam (310 mm PU)",
    wall_material="aluminium", tech_year=2021, aircraft_class="reference tank", conditions="max pressure 404 kPa; L=9.3 m cylinder + hemispheres; 5% ullage",
    notes="Tank mass 3079 kg.", confidence="high")
add(id="HU21-100m3-MLI", source_key="huete2021", year_published=2021, data_type="design_study", gi_scope="tank", gi=0.643,
    m_h2_kg_per_tank=6128, m_h2_basis="stated", n_tanks=1, diameter_m=4.0, volume_m3=100, insulation="MLI vacuum (127 mm), stiffened panels",
    wall_material="aluminium", tech_year=2021, aircraft_class="reference tank", conditions="max pressure 210 kPa",
    notes="Tank mass 3554 kg.", confidence="high")
add(id="HU21-sphere-MLI", source_key="huete2021", year_published=2021, data_type="design_study", gi_scope="tank", gi=0.744,
    m_h2_kg_per_tank=6405, m_h2_basis="stated", n_tanks=1, diameter_m=5.76, volume_m3=100, insulation="MLI vacuum",
    wall_material="aluminium", tech_year=2021, aircraft_class="reference tank", conditions="sphere r=2.88 m; 210 kPa",
    notes="Tank mass 2199 kg. Upper bound on shape efficiency at this volume.", confidence="high")
add(id="HU21-0p5m", source_key="huete2021", year_published=2021, data_type="design_study", gi_scope="tank", gi=0.10,
    m_h2_kg_per_tank=15, m_h2_basis="estimated", diameter_m=0.5, insulation="best of foam/MLI", wall_material="aluminium", tech_year=2021,
    aircraft_class="very small", notes="Text: 'as low as 0.1 for 0.5 m tanks'. Mass estimated for L/D~3 (0.3 m3). Figure-read precision only.",
    confidence="medium (GI) / low (mass)")
add(id="HU21-900m3", source_key="huete2021", year_published=2021, data_type="design_study", gi_scope="tank", gi=0.75, gi_low=0.70, gi_high=0.80,
    m_h2_kg_per_tank=54000, m_h2_basis="estimated", diameter_m=7.0, volume_m3=900, insulation="foam", wall_material="aluminium", tech_year=2021,
    aircraft_class="very large", notes="Text: 'up to nearly 0.8 for very large tanks'; foam ~0.70+ at 900 m3. Mass = 0.85 fill x 71 kg/m3 x 900 m3.",
    confidence="low")
add(id="HU21-300m3", source_key="huete2021", year_published=2021, data_type="design_study", gi_scope="tank", gi=0.70, gi_low=0.66, gi_high=0.74,
    m_h2_kg_per_tank=18000, m_h2_basis="estimated", diameter_m=6.0, volume_m3=300, insulation="foam", wall_material="aluminium", tech_year=2021,
    aircraft_class="large", notes="Text: tanks for ranges above regional give >0.66; foam superior to vacuum at 300 m3. GI interpolated between 100 m3 and 900 m3 values.",
    confidence="low")

# ---------------- Huete, Nalianda, Pilidis 2022 (Aeronautical Journal) ----------------
for v, yr, lab, conf in [(0.45,2022,"baseline (current conventional materials)","medium"),(0.60,2035,"first-generation EIS","medium"),
                         (0.70,2036,"feasible for EIS in 12-15 years (2034-2037)","medium"),(0.75,2050,"second/third generation","low")]:
    add(id=f"HU22-{v}", source_key="huete2022", year_published=2022, data_type="projection", gi_scope="tank", gi=v,
        m_h2_kg_per_tank=6128, m_h2_basis="estimated", volume_m3=100, diameter_m=4.0, tech_year=yr, aircraft_class="widebody family (HVL*)",
        notes=f"{lab}. Evaluated on 100 m3 reference tank; tech_year mapping for 2nd/3rd gen is my estimate.", confidence=conf)

# ---------------- Verstraete et al. 2010 (IJHE) ----------------
for lab, gi, mt, mpt, n, d in [("single PU foam",0.709,473.4,1150,1,3.0),("single Rohacell",0.686,526.0,1150,1,3.0),
                               ("single MLI",0.675,555.0,1150,1,3.0),("twin 3.0 m",0.658,299.9,575,2,3.0),
                               ("twin 2.5 m",0.653,306.2,575,2,2.5),("twin 2.0 m",0.619,354.3,575,2,2.0)]:
    add(id=f"VE10-reg-{lab.replace(' ','_')}", source_key="verstraete2010", year_published=2010, data_type="design_study", gi_scope="tank",
        gi=gi, m_h2_kg_per_tank=mpt, m_h2_basis="derived", n_tanks=n, m_h2_kg_aircraft=1150, diameter_m=d,
        insulation="MLI" if "MLI" in lab else "foam", wall_material="aluminium", tech_year=2010, aircraft_class="regional (32 pax, 2100 km)",
        conditions="integral tank; no hold period", notes=f"Tank mass {mt} kg. With a 2 h pre-takeoff hold GI drops ~4 points.", confidence="high")
for fuel, gi, mt in [(15000,0.764,4667),(20000,0.769,6018),(25000,0.780,7083),(30000,0.780,8491),(35000,0.785,9612)]:
    add(id=f"VE10-LR-{fuel//1000}t", source_key="verstraete2010", year_published=2010, data_type="design_study", gi_scope="tank",
        gi=gi, m_h2_kg_per_tank=fuel, m_h2_basis="stated", n_tanks=1, m_h2_kg_aircraft=fuel, diameter_m=7.5,
        insulation="foam", wall_material="aluminium", tech_year=2010, aircraft_class="very large LR (400-550 pax, 7500 nm)",
        conditions="integral tank", notes=f"Tank mass {mt} kg. n_tanks=1 inferred from tank length vs volume; diameter ~7.6-8.5 m fuselage.",
        confidence="high (GI) / medium (per-tank)")

# ---------------- Jagtap et al. 2023 literature table (secondary) ----------------
for k, gi, lo, hi, cls, note in [
    ("NACA1955", 0.88, "", "", "high-altitude reconnaissance", "Integral, stainless steel + foam"),
    ("Brewer-nonintegral", 0.90, "", "", "generic cylindrical", "Brewer 1991 / Verstraete 2009, non-integral Al + foam"),
    ("Gomez2019", 0.785, 0.74, 0.83, "long-range 197 pax", "Gomez & Smith 2019, integral Al + PU"),
    ("Brewer1991-LR", 0.84, "", "", "long-range 400 pax", "Brewer 1991 / Gomez 2019, integral Al + foam"),
]:
    add(id=f"JA23-{k}", source_key="jagtap2023", year_published=2023, data_type="design_study", gi_scope="tank", gi=gi, gi_low=lo, gi_high=hi,
        insulation="foam", wall_material="aluminium or steel", tech_year="", aircraft_class=cls,
        notes=f"Secondary (Jagtap Table SI 1): {note}. No mass given; use as ceiling evidence only.", confidence="medium")

# ---------------- McKinsey / Clean Sky 2 & FCH JU 2020 ----------------
# EIS windows counted from 2020. LH2 per aircraft estimated by scaling FlyZero design-mission fuel with pax x range.
for cls, gi, eis, lo_e, mest in [("commuter 19 pax 500 km",0.25,2028,"<10 y",80),("regional 80 pax 1000 km",0.30,2033,"10-15 y",800),
                                 ("short-range 165 pax 2000 km",0.35,2035,"15 y",2500),("medium-range 250 pax 7000 km",0.37,2040,"20 y",10000),
                                 ("long-range 325 pax 10000 km",0.38,2043,"20-25 y",19000)]:
    add(id=f"MK20-{cls.split()[0]}", source_key="mckinsey2020", year_published=2020, data_type="target", gi_scope="system",
        gi=gi, m_h2_kg_per_tank=mest/2, m_h2_basis="estimated", n_tanks=2, m_h2_kg_aircraft=mest, insulation="vacuum/MLI (double wall)",
        wall_material="", tech_year=eis, aircraft_class=cls,
        notes=f"Target GI in report exhibits; EIS {lo_e} from 2020. LH2 mass is my estimate (scaled from FlyZero design-mission fuel). 12 kWh/kg ~ GI 35%.",
        confidence="high (GI) / low (mass, year)")
add(id="MK20-state2020", source_key="mckinsey2020", year_published=2020, data_type="hardware_claim", gi_scope="system", gi=0.20,
    m_h2_kg_per_tank=40, m_h2_basis="estimated", tech_year=2020, aircraft_class="commuter",
    notes="'Latest concepts for commuter aircraft have a gravimetric index of up to 20 percent'. Report also asks for 50% tank mass cut vs current prototypes.",
    confidence="medium")

# ---------------- Roadmaps, requirements, targets ----------------
add(id="CAW21-state", source_key="ca_workshop2021", year_published=2021, data_type="hardware_claim", gi_scope="unclear", gi=0.26, gi_low=0.25, gi_high=0.27,
    tech_year=2021, aircraft_class="regional/SR", notes="'GI that can be reached (without any technological breakthrough) is 25-27%'. Al welded, vacuum insulated tanks.", confidence="medium")
add(id="CAW21-35", source_key="ca_workshop2021", year_published=2021, data_type="target", gi_scope="unclear", gi=0.35,
    tech_year=2035, aircraft_class="SR", notes="35% from the McKinsey study 'considered feasible while the definition of the conditions is critical'.", confidence="medium")
add(id="IATA23-30", source_key="iata2023", year_published=2023, data_type="projection", gi_scope="unclear", gi=0.30,
    tech_year=2035, aircraft_class="narrowbody", notes="Milestone 'High gravimetric index of LH2 tanks demonstrated (GI > 30%)' placed ~2030-2040 on roadmap; lower bound, not point value.", confidence="medium (GI) / low (year)")
add(id="ICCT22-range", source_key="icct2022", year_published=2022, data_type="projection", gi_scope="system", gi=0.275, gi_low=0.20, gi_high=0.35,
    tech_year=2035, aircraft_class="regional to narrowbody", notes="Achievable fuel-system GI range swept by ICCT for 2035 evolutionary aircraft; 0.35 = mass-parity point.", confidence="medium")
add(id="MIT23-base", source_key="mit2023", year_published=2023, data_type="projection", gi_scope="tank", gi=0.35, tech_year=2035,
    aircraft_class="regional (Dash 8-400 retrofit)", notes="Baseline assumption. Improved case 0.50.", confidence="medium")
add(id="MIT23-improved", source_key="mit2023", year_published=2023, data_type="projection", gi_scope="tank", gi=0.50, tech_year=2040,
    aircraft_class="regional (Dash 8-400 retrofit)", notes="Improved case; tech_year my estimate.", confidence="low (year)")
add(id="MIT23-shuttle-PRSA", source_key="mit2023", year_published=2023, data_type="hardware_measured", gi_scope="tank", gi=0.25,
    m_h2_kg_per_tank=100, m_h2_basis="stated", tech_year=1981, aircraft_class="spacecraft (Shuttle on-board)",
    notes="'25% has been achieved in shuttle on-board LH2 storage of 100 kg'. Long-dormancy supercritical storage; small-size hardware anchor.", confidence="medium")
add(id="MIT23-space-large", source_key="mit2023", year_published=2023, data_type="hardware_measured", gi_scope="tank", gi=0.835, gi_low=0.83, gi_high=0.84,
    m_h2_kg_per_tank=100000, m_h2_basis="estimated", tech_year=2000, aircraft_class="launcher (Shuttle ET class)",
    notes="'83-84% achieved in large-scale space applications'. Minutes of dormancy; ceiling anchor only.", confidence="medium")
add(id="AM23-evol", source_key="adler2023", year_published=2023, data_type="projection", gi_scope="tank", gi=0.325, gi_low=0.25, gi_high=0.40,
    tech_year=2035, aircraft_class="generic", notes="'Evolutionary improvements predicted 25%-40%'.", confidence="medium")
add(id="AM23-revol", source_key="adler2023", year_published=2023, data_type="projection", gi_scope="tank", gi=0.70, gi_low=0.70,
    tech_year=2050, aircraft_class="generic", notes="'Revolutionary improvements 70% or more'; also tipping point ~55% for H2 to beat kerosene with range.", confidence="medium")
add(id="TI24-70", source_key="tiwari2024", year_published=2024, data_type="projection", gi_scope="tank", gi=0.70, tech_year=2035,
    aircraft_class="SR-LR", notes="'A cryogenic fuel tank gravimetric efficiency of 70% or above is expected by 2035' (relies on FlyZero).", confidence="low")
add(id="ONERA24-SA", source_key="onera2024", year_published=2024, data_type="design_study", gi_scope="system", gi=0.35,
    m_h2_kg_per_tank=4575, m_h2_basis="derived", n_tanks=2, m_h2_kg_aircraft=9150, volume_m3=60, insulation="vacuum + MLI",
    wall_material="Al 6061 inner, light alloy/composite outer", tech_year=2035, aircraft_class="single aisle (GRAVITHY)",
    conditions="12 h dormancy; 2 bar op, 4 bar vent; GI incl. distribution", notes="Assumed GI; tank mass ~17 t for ~120 m3 -> m_H2 derived as 0.35/0.65 x 17 t.", confidence="medium")
add(id="CAcall26-FTA07", source_key="ca_call_fta07", year_published=2026, data_type="requirement", gi_scope="tank", gi=0.40, gi_low=0.40,
    m_h2_kg_per_tank=800, m_h2_basis="stated", diameter_m=2.5, insulation="", tech_year=2028, aircraft_class="regional/SR",
    conditions="TRL4; dormancy >= 12 h; ~600 kg (fuel cell) or ~1000 kg (combustion) per tank; D 1.5-3.5 m",
    notes="Clean Aviation call HORIZON-JU-CLEAN-AVIATION-2026-04-FTA-07 (max 24 months): GI 'no less than 40%'. Mass = midpoint of 600/1000.", confidence="high")
add(id="CAcall26-HPA02", source_key="ca_call_hpa02", year_published=2026, data_type="requirement", gi_scope="system", gi=0.30, gi_low=0.30,
    m_h2_kg_per_tank=600, m_h2_basis="estimated", tech_year=2029, aircraft_class="regional fuel cell",
    notes="GI > 30% for entire onboard storage, supply and conditioning system (tank + piping + venting + conditioning).", confidence="high (GI) / low (mass)")
add(id="OVERLEAF-SoA", source_key="overleaf", year_published=2022, data_type="hardware_claim", gi_scope="tank", gi=0.20,
    m_h2_kg_per_tank=500, m_h2_basis="stated", tech_year=2022, aircraft_class="SR/regional",
    notes="Project baseline: current technology achieves only 20% GI; industry needs >= 35%.", confidence="medium")
add(id="OVERLEAF-target", source_key="overleaf", year_published=2022, data_type="target", gi_scope="tank", gi=0.60, gi_low=0.60,
    m_h2_kg_per_tank=500, m_h2_basis="stated", insulation="open-cell / low-pressure", wall_material="thermoplastic composite",
    tech_year=2025, aircraft_class="SR/regional", notes="Target >60% for 500 kg; project ended Oct 2025, TRL6 targeted 2028; achieved GI not published.", confidence="high (target) / unknown (achieved)")

# ---------------- Hardware (measured or vendor-claimed) ----------------
add(id="COCO-demo1", source_key="cocolih2t2026", year_published=2026, data_type="hardware_measured", gi_scope="tank", gi=0.16,
    m_h2_kg_per_tank=57, m_h2_basis="stated", volume_m3=1.1, insulation="vacuum", wall_material="thermoplastic/thermoset composite, linerless",
    tech_year=2025, aircraft_class="regional (ATR72-like tailcone)", conditions="conformal; dormancy >24 h target",
    notes="Estimated GI of Demonstrator 1 incl. SS pipes and interface plates (hardware-based estimate).", confidence="high")
add(id="COCO-target", source_key="cocolih2t2026", year_published=2026, data_type="target", gi_scope="tank", gi=0.25,
    m_h2_kg_per_tank=57, m_h2_basis="stated", volume_m3=1.1, insulation="vacuum", wall_material="composite", tech_year=2026,
    aircraft_class="regional", notes="Project KPI, TRL4.", confidence="high")
add(id="GTL-19kg", source_key="gtl2024", year_published=2024, data_type="hardware_claim", gi_scope="tank", gi=0.559,
    m_h2_kg_per_tank=19, m_h2_basis="stated", diameter_m=0.71, insulation="vacuum dewar", wall_material="composite",
    tech_year=2024, aircraft_class="UAV/small", conditions="~1%/day boil-off; 21 h hold tested",
    notes="Vendor claim: 15 kg tank holding 19 kg LH2 (28 x 53 in).", confidence="medium (vendor)")
add(id="GTL-50kg", source_key="gtl2024", year_published=2024, data_type="hardware_claim", gi_scope="tank", gi=0.625,
    m_h2_kg_per_tank=50, m_h2_basis="stated", insulation="vacuum dewar", wall_material="composite", tech_year=2024, aircraft_class="small",
    notes="Vendor claim: 30 kg tank for 50 kg LH2; 'larger versions able to achieve over 70%'.", confidence="medium (vendor)")
add(id="UAV25-12L", source_key="uav2025", year_published=2025, data_type="hardware_measured", gi_scope="tank", gi=0.24,
    m_h2_kg_per_tank=0.88, m_h2_basis="derived", volume_m3=0.012, insulation="vacuum", wall_material="titanium", tech_year=2025,
    aircraft_class="UAV", notes="Five 12 L tanks at ~24%; empty mass 2.8 kg; 23-50 h total evaporation.", confidence="high")
add(id="HALE09-est", source_key="nasa_hale2009", year_published=2009, data_type="design_study", gi_scope="tank", gi=0.672,
    m_h2_kg_per_tank=600, m_h2_basis="derived", n_tanks=2, diameter_m=2.59, insulation="vacuum/MLI (long endurance)",
    tech_year=2009, aircraft_class="HALE UAV", notes="645 lb estimated tank for 1323 lb LH2 per spherical tank; 10-16 day mission.", confidence="medium")
add(id="HALE09-goal", source_key="nasa_hale2009", year_published=2009, data_type="target", gi_scope="tank", gi=0.807,
    m_h2_kg_per_tank=600, m_h2_basis="derived", n_tanks=2, diameter_m=2.59, tech_year=2015, aircraft_class="HALE UAV",
    notes="Advanced-technology dry-mass goal 316 lb per tank.", confidence="medium")

# ---------------- Other design studies ----------------
add(id="SA22-100kg", source_key="saias2022", year_published=2022, data_type="design_study", gi_scope="tank", gi=0.26,
    m_h2_kg_per_tank=100, m_h2_basis="stated", insulation="MLI", wall_material="Al 2014 liner / SS 304L outer", tech_year=2022,
    aircraft_class="rotorcraft", conditions="1.45 bar, 0.1%/h boil-off", notes="'26-34% for 100-500 kg'; endpoints paired with endpoints.", confidence="medium")
add(id="SA22-500kg", source_key="saias2022", year_published=2022, data_type="design_study", gi_scope="tank", gi=0.34,
    m_h2_kg_per_tank=500, m_h2_basis="stated", insulation="MLI", wall_material="Al 2014 liner / SS 304L outer", tech_year=2022,
    aircraft_class="rotorcraft", conditions="1.45 bar", notes="See SA22-100kg.", confidence="medium")
add(id="DLR26-range", source_key="dlr2026", year_published=2026, data_type="design_study", gi_scope="system", gi=0.353, gi_low=0.2536, gi_high=0.4521,
    m_h2_kg_per_tank=1225, m_h2_basis="stated", tech_year=2040, aircraft_class="short-range fuel cell",
    conditions="surrogate over 750-1700 kg per tank, 2.2-3.0 m internal height, 1.5-3.0 bar venting",
    notes="3570 converged designs; GI range spans all design parameters, not size alone. GI best near 2.0 bar vent. Full paper needs your approval to fetch (mdpi.com).",
    confidence="medium")
add(id="LI23-345kg", source_key="licheva2023", year_published=2023, data_type="design_study", gi_scope="tank", gi=0.19,
    m_h2_kg_per_tank=345, m_h2_basis="stated", diameter_m=2.0, insulation="foam", wall_material="Al 2219", tech_year=2023,
    aircraft_class="regional", conditions="10 bar venting, ~10 h exposure", notes="Tank+insulation ~1450-1500 kg (read from multi-tank case). Conservative long-exposure case.",
    confidence="low")
for m, gi in [(70.45,0.529),(90.35,0.517)]:
    add(id=f"MZ24-{m}", source_key="mazzoni2024", year_published=2024, data_type="design_study", gi_scope="tank", gi=gi,
        m_h2_kg_per_tank=m, m_h2_basis="stated", insulation="MLI Mylar/Dacron", tech_year=2024, aircraft_class="regional (ATR-based, Leonardo)",
        notes="Optimized LH2 storage case study.", confidence="medium")
add(id="MZ24-UAV-ref", source_key="mazzoni2024", year_published=2024, data_type="design_study", gi_scope="tank", gi=0.264,
    m_h2_kg_per_tank=2.26, m_h2_basis="stated", tech_year=2024, aircraft_class="UAV", notes="Reference config, 6.3 kg empty tank.", confidence="medium")
add(id="MZ24-UAV-MLI", source_key="mazzoni2024", year_published=2024, data_type="design_study", gi_scope="tank", gi=0.319,
    m_h2_kg_per_tank=2.26, m_h2_basis="stated", insulation="MLI", tech_year=2024, aircraft_class="UAV", notes="Optimized.", confidence="medium")
add(id="SV23-ATR42", source_key="svensson2023", year_published=2023, data_type="design_study", gi_scope="tank", gi=0.535,
    m_h2_kg_per_tank=198, m_h2_basis="derived", n_tanks=2, m_h2_kg_aircraft=396, volume_m3=3.0, tech_year=2023, aircraft_class="regional (ATR42-600)",
    notes="6 m3 H2 in two tanks, tank weight 342.4 kg total; m_H2 = 6 m3 x 71 x 0.93.", confidence="medium")
add(id="SV23-single-range", source_key="svensson2023", year_published=2023, data_type="design_study", gi_scope="tank", gi=0.41, gi_low=0.32, gi_high=0.50,
    m_h2_kg_per_tank=700, m_h2_basis="estimated", volume_m3=11, tech_year=2023, aircraft_class="regional",
    notes="Single tank GI ~0.32-0.50 over 2-20 m3 (vent pressure dependent). Not a clean size sweep.", confidence="low")
add(id="WI18-cyl", source_key="winnefeld2018", year_published=2018, data_type="design_study", gi_scope="tank", gi=0.66, gi_low=0.64, gi_high=0.68,
    m_h2_kg_per_tank=1912, m_h2_basis="stated", insulation="foam", wall_material="aluminium", tech_year=2018, aircraft_class="short/medium range",
    notes="Single cylindrical tank (phi=1). 4 tanks instead of 1: ~0.50-0.55.", confidence="medium")
add(id="WI18-sphere", source_key="winnefeld2018", year_published=2018, data_type="design_study", gi_scope="tank", gi=0.70,
    m_h2_kg_per_tank=1912, m_h2_basis="stated", insulation="foam", wall_material="aluminium", tech_year=2018, aircraft_class="short/medium range",
    notes="Spherical single tank.", confidence="medium")
add(id="WI18-4tanks", source_key="winnefeld2018", year_published=2018, data_type="design_study", gi_scope="tank", gi=0.525, gi_low=0.50, gi_high=0.55,
    m_h2_kg_per_tank=478, m_h2_basis="derived", n_tanks=4, m_h2_kg_aircraft=1912, insulation="foam", wall_material="aluminium", tech_year=2018,
    aircraft_class="short/medium range", notes="Same fuel split over 4 cylindrical tanks: direct evidence of per-tank size effect.", confidence="medium")
add(id="IZ25-tank", source_key="izea2025", year_published=2025, data_type="design_study", gi_scope="tank", gi=0.67,
    m_h2_kg_per_tank=875, m_h2_basis="stated", n_tanks=2, m_h2_kg_aircraft=1750, volume_m3=13, insulation="foam (5.3 cm PU)", wall_material="Al 2219",
    tech_year=2025, aircraft_class="BWB >100 pax", conditions="vent 1.36-1.74 bar", notes="Optimal tank-only GI.", confidence="high")
add(id="IZ25-system", source_key="izea2025", year_published=2025, data_type="design_study", gi_scope="system", gi=0.62,
    m_h2_kg_per_tank=875, m_h2_basis="stated", n_tanks=2, m_h2_kg_aircraft=1750, volume_m3=13, insulation="foam", wall_material="Al 2219",
    tech_year=2025, aircraft_class="BWB >100 pax", notes="Incl. heat exchangers and thermal management.", confidence="high")
for lab, gi, m, vol, mat_ in [("allmetal",0.2931,6400,100,"metal (Ti outer)"),("composite",0.5011,6400,100,"sandwich composite"),
                              ("composite-real",0.4922,6400,100,"sandwich composite + realistic features"),
                              ("aft55",0.4860,3520,55,"sandwich composite"),("fwd37",0.4841,2370,37,"sandwich composite")]:
    add(id=f"SO25-{lab}", source_key="soton2025", year_published=2025, data_type="design_study", gi_scope="tank", gi=gi,
        m_h2_kg_per_tank=m, m_h2_basis="estimated", volume_m3=vol, insulation="passive (sandwich core)", wall_material=mat_, tech_year=2025,
        aircraft_class="SR/MR", conditions="2.5%/day boil-off", notes="Mass = 0.9 fill x 71 x volume. Pressure-vessel sized; weak size effect between 37 and 100 m3.",
        confidence="high (GI) / low (mass)")
add(id="AE24-465kg", source_key="aerospace2024", year_published=2024, data_type="design_study", gi_scope="tank", gi=0.41,
    m_h2_kg_per_tank=465, m_h2_basis="stated", tech_year=2024, aircraft_class="small/regional", notes="Tank mass 674 kg. Aerospace 2024, 11(2), 161.", confidence="medium")
add(id="PA24-range", source_key="parello2024", year_published=2024, data_type="design_study", gi_scope="tank", gi=0.70, gi_low=0.55, gi_high=0.85,
    m_h2_kg_per_tank=4000, m_h2_basis="estimated", tech_year=2024, aircraft_class="A320-class",
    notes="ISAE/ONERA DoE: adjusted GI 0.55-0.85; 7 cm insulation: 0.78 low pressure, 0.60 high pressure. Mass estimated.", confidence="low (mass)")
add(id="CH22-2050LR", source_key="chalmers2022", year_published=2022, data_type="design_study", gi_scope="tank", gi=0.70, gi_low=0.60, gi_high=0.74,
    m_h2_kg_per_tank=17500, m_h2_basis="derived", insulation="rigid PVC foam", tech_year=2050, aircraft_class="long-range (A350-1000 class)",
    notes="G~0.7 excl. fairing, ~0.6 incl. fairing; tank+insulation = 35% of fuel mass. Study also states G>=0.5 required for long range.", confidence="medium")

cols = ["id","source_key","year_published","data_type","gi_scope","gi","gi_low","gi_high","m_h2_kg_per_tank","m_h2_basis","n_tanks",
        "m_h2_kg_aircraft","diameter_m","volume_m3","insulation","wall_material","tech_year","aircraft_class","conditions","notes","confidence","url"]
with open("lh2_tank_gi_dataset.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({c: r.get(c,"") for c in cols})
print(len(rows), "rows")
