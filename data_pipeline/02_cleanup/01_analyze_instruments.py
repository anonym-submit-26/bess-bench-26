#!/usr/bin/env python3
"""
Analysis and standardization of BeSS instruments.

Phase 1: Generates instrument_lookup.json from unique fits_bss_inst values.
Phase 2: 02_apply_instrument_cleanup.py applies it to the database.

Approach: text normalization + regex patterns + professional lookups.
"""

import re
import json
import duckdb
from pathlib import Path
from collections import Counter


# =============================================================================
# NORMALISATION
# =============================================================================

def normalize(text):
    """Normalise the text for matching: _ - + → spaces."""
    return re.sub(r'[_\-+]', ' ', text)


def S(pattern, text, **kw):
    """Shortcut: re.search case-insensitive."""
    return re.search(pattern, text, re.IGNORECASE | kw.get('flags', 0))



# =============================================================================
# PARSE TELESCOPE
# =============================================================================

def parse_telescope(raw):
    """Extract the telescope from the fits_bss_inst text."""
    text = normalize(raw)  # _ - + → spaces
    
    # === Formats @ (professionnels) ===
    m = S(r'@OHP\s*193', text)
    if m: return 'OHP_T193'
    m = S(r'@OHP', text)
    if m: return 'OHP'
    m = S(r'@TBL', text)
    if m: return 'PicDuMidi_TBL'
    m = S(r'@TNG', text)
    if m: return 'TNG_3.6m'
    m = S(r'@ESO\s*([\d.]+)', text)
    if m: return f'ESO_{m.group(1)}m'
    m = S(r'Feros@ESO([\d.]+)', raw)
    if m: return f'ESO_{m.group(1)}m'
    
    # === Celestron / Schmidt-Cassegrain ===
    # Pattern: C14, C11, C10, C9.25, C9, C8, C5 — followed by non-digit
    m = S(r'(?:Celestron\s*)?C\s*14(?=[^0-9]|$)', text)
    if m: return 'C14'
    m = S(r'(?:Celestron\s*)?C\s*11(?=[^0-9]|$)', text)
    if m: return 'C11'
    m = S(r'(?:Celestron\s*)?C\s*10(?=[^0-9]|$)', text)
    if m: return 'C10'
    m = S(r'(?:Celestron\s*)?C\s*9\.?25(?=[^0-9]|$)', text)
    if m: return 'C9'
    m = S(r'(?:Celestron\s*)?C\s*9(?=[^0-9]|$)', text)
    if m: return 'C9'
    m = S(r'(?:Celestron\s*)?C\s*8(?=[^0-9]|$)', text)
    if m: return 'C8'
    m = S(r'(?:Celestron\s*)?C\s*5(?=[^0-9]|$)', text)
    if m: return 'C5'
    # Edge HD variants : "C8 Edge", "C8Edge", "EdgeHD8", "EdgeHD 8", "14 Edge HD"
    m = S(r'Edge\s*(?:HD)?\s*(\d+)', text)
    if m: return f'C{m.group(1)}'
    m = S(r'(\d+)\s*(?:inch|in|")?\s*Edge\s*HD', text)
    if m: return f'C{m.group(1)}'
    # SC prefix : SC14, SC12, SC25, SC8
    m = S(r'SC\s*(\d+)(?=[^0-9]|$)', text)
    if m:
        val = int(m.group(1))
        if val <= 25:  # C-series inch sizes
            return f'SC{val}' if val > 14 else f'C{val}'
    # SCT + meters: SCT0.3m
    m = S(r'SCT\s*0?\.(\d+)\s*m', text)
    if m: return f'SCT_{m.group(1)}cm'
    
    # === Ritchey-Chretien / RC ===
    m = S(r'RC\s*(\d{2,3})(?=[^0-9]|$)', text)
    if m:
        val = int(m.group(1))
        return f'RC{val}'
    m = S(r'RCO\s*(\d+)', text)
    if m: return f'RC{m.group(1)}'
    m = S(r'RCX\s*(\d+)', text)
    if m: return f'RC{m.group(1)}'
    # Astrosib (often with RC)
    m = S(r'Astrosib', text)
    if m and not S(r'RC', text):
        return 'Astrosib'
    
    # === CDK (Corrected Dall-Kirkham) ===
    m = S(r'CDK\s*(\d+)', text)
    if m: return f'CDK{m.group(1)}'
    # "CDK" alone (without number)
    m = S(r'(?<![a-z])CDK(?=[^0-9a-z]|$)', text)
    if m: return 'CDK'
    
    # === Takahashi ===
    m = S(r'FSQ\s*106', text)
    if m: return 'Takahashi_FSQ106'
    m = S(r'FSQ\s*85', text)
    if m: return 'Takahashi_FSQ85'
    m = S(r'FS\s*128', text)
    if m: return 'Takahashi_FS128'
    m = S(r'TSA\s*120', text)
    if m: return 'Takahashi_TSA120'
    m = S(r'Mewlon\s*210', text)
    if m: return 'Takahashi_Mewlon210'
    m = S(r'Mewlon\s*180', text)
    if m: return 'Takahashi_Mewlon180'
    m = S(r'R\s*128', text)
    if m: return 'Takahashi_R128'
    
    # === Vixen ===
    m = S(r'VC\s*200\s*L?', text)
    if m: return 'Vixen_VC200L'
    m = S(r'VMC\s*200', text)
    if m: return 'Vixen_VMC200'
    m = S(r'R200SS', text)
    if m: return 'Vixen_R200SS'
    
    # === Newton / Newtonian ===
    # NEWTON250, Newton 200, Newt200, NW254, Nwt200, N200, N250, N300, N350
    m = S(r'(?:NEWTON|Newton|Newt|NW|Nwt)\s*(\d{2,4})', text)
    if m:
        val = int(m.group(1))
        if val < 30:  # inches
            return f'Newton_{val}inch'
        return f'Newton_{val}mm'
    # N+diameter but NOT NOU, NOU_T, Neo, Nova, Nikon
    m = S(r'(?<![a-z])N\s*(\d{3})(?=[^0-9]|$)', text)
    if m and not S(r'NO[UV]|Neo|Nova|Nik', text):
        return f'Newton_{m.group(1)}mm'
    # "10in Newt" etc.
    m = S(r'(\d+)\s*(?:in|inch)?\s*(?:Newt|Newton)', text)
    if m:
        val = int(m.group(1))
        if val < 30:
            return f'Newton_{val}inch'
        return f'Newton_{val}mm'
    
    # === Meade / LX ===
    m = S(r'LX\s*200', text)
    if m: return 'Meade_LX200'
    m = S(r'LX\s*254', text)
    if m: return 'Meade_LX254'
    m = S(r'LX\s*90', text)
    if m: return 'Meade_LX90'
    m = S(r'MEADE\s*(\d+)\s*mm', text)
    if m: return f'Meade_{m.group(1)}mm'
    m = S(r'MEADE\s*(\d+)', text)
    if m:
        val = int(m.group(1))
        if val <= 20: return f'Meade_{val}inch'
        return f'Meade_{val}mm'
    
    # === Telescopes T+diameter (French) ===
    # T60, T62, T80, T120, T150, T190, T193, T200...T1000
    # Note: T60 = PicDuMidi, T80 = OHP, T193 = OHP, T152 = OHP
    m = S(r'(?<![a-z])T\s*1000(?=[^0-9]|$)', text)
    if m: return 'T1000'
    m = S(r'(?<![a-z])T\s*600(?=[^0-9]|$)', text)
    if m: return 'T600'
    m = S(r'(?<![a-z])T\s*520(?=[^0-9]|$)', text)
    if m: return 'T520'
    m = S(r'(?<![a-z])T\s*500(?=[^0-9]|$)', text)
    if m: return 'T500'
    m = S(r'(?<![a-z])T\s*400(?=[^0-9]|$)', text)
    if m: return 'T400'
    m = S(r'(?<![a-z])T\s*350(?=[^0-9]|$)', text)
    if m: return 'T350'
    m = S(r'(?<![a-z])T\s*300(?=[^0-9]|$)', text)
    if m: return 'T300'
    m = S(r'(?<![a-z])T\s*254(?=[^0-9]|$)', text)
    if m: return 'T254'
    m = S(r'(?<![a-z])T\s*250(?=[^0-9]|$)', text)
    if m: return 'T250'
    m = S(r'(?<![a-z])T\s*200(?=[^0-9]|$)', text)
    if m: return 'T200'
    m = S(r'(?<![a-z])T\s*193(?=[^0-9]|$)', text)
    if m: return 'OHP_T193'
    m = S(r'(?<![a-z])T\s*190(?=[^0-9]|$)', text)
    if m: return 'T190'
    m = S(r'(?<![a-z])T\s*152(?=[^0-9]|$)', text)
    if m: return 'OHP_T152'
    m = S(r'(?<![a-z])T\s*150(?=[^0-9]|$)', text)
    if m: return 'T150'
    m = S(r'(?<![a-z])T\s*120(?=[^0-9]|$)', text)
    if m: return 'OHP_T120'
    m = S(r'(?<![a-z])T\s*80(?=[^0-9]|$)', text)
    if m: return 'OHP_T80'
    m = S(r'(?<![a-z])T\s*62(?=[^0-9]|$)', text)
    if m: return 'OHP_T62'
    m = S(r'(?<![a-z])T\s*60(?=[^0-9]|$)', text)
    if m: return 'PicDuMidi_T60'
    # T0.18 → T180mm
    m = S(r'(?<![a-z])T\s*0\.(\d+)', text)
    if m:
        frac = m.group(1)
        mm = int(frac) * (100 if len(frac) == 1 else 10 if len(frac) == 2 else 1)
        return f'T{mm}mm'
    
    # === Special Observatories ===
    m = S(r'IAC\s*80', text)
    if m: return 'IAC80'
    m = S(r'C2PU', text)
    if m: return 'C2PU_T1000'
    m = S(r'Pic\s*of the\s*Midi', text)
    if m: return 'PicDuMidi_T60'
    # GEMINI
    m = S(r'GEMINI', text)
    if m: return 'Gemini'
    
    # === DK (Dall-Kirkham) ===
    m = S(r'(\d+)\s*cm\s*DK', raw)  # 31cmDK
    if m: return f'DK_{m.group(1)}cm'
    m = S(r'DK\s*(\d+)', text)
    if m:
        val = int(m.group(1))
        if val > 100: return f'DK_{val}mm'
        return f'DK_{val}cm'
    
    # === CEDES ===
    m = S(r'CEDES\s*(\d+)', text)
    if m: return f'CEDES_{m.group(1)}cm'
    
    # === Various Manufacturers ===
    # Askar
    m = S(r'Askar\s*(\d+)', text)
    if m: return f'Askar_{m.group(1)}'
    # AP (Astro-Physics)
    m = S(r'AP\s*130', text)
    if m: return 'AP130'
    m = S(r'AP\s*115', text)
    if m: return 'AP115'
    # Esprit
    m = S(r'Esprit\s*(\d+)', text)
    if m: return f'Esprit_{m.group(1)}'
    # APM
    m = S(r'APM\s*(\d+)', text)
    if m: return f'APM_{m.group(1)}'
    # Mak
    m = S(r'MAK\s*(\d+)', text)
    if m: return f'Mak{m.group(1)}'
    # MTO
    m = S(r'MTO\s*(\d+)', text)
    if m: return f'MTO_{m.group(1)}'
    # FRA
    m = S(r'FRA\s*(\d+)', text)
    if m: return f'FRA_{m.group(1)}'
    # EB1, EB2, EB3
    m = S(r'EB\s*([123])', text)
    if m: return f'EB{m.group(1)}'
    # CN212
    m = S(r'CN\s*212', text)
    if m: return 'CN212'
    # CC13
    m = S(r'CC\s*13', text)
    if m: return 'CC13'
    # OMC
    m = S(r'OMC\s*(\d+)', text)
    if m: return f'OMC{m.group(1)}'    # NP101
    m = S(r'NP\\s*101', text)
    if m: return 'NP101'    # TSC254, TSC225
    m = S(r'TSC\s*(\d+)', text)
    if m: return f'TSC_{m.group(1)}'
    # Perl
    m = S(r'Perl', text)
    if m: return 'Perl'
    # SkyWatcher / SW (Note: SW is ambiguous, but in BeSS it is a telescope)
    m = S(r'Sky\s*Watcher\s*(\d+)?', text)
    if m: return f'SkyWatcher_{m.group(1)}' if m.group(1) else 'SkyWatcher'
    m = S(r'(?<![a-z])SW\s*(\d+)', text)
    if m and not S(r'SWP|SX', raw):  # not IUE_SWP or SXVR
        return f'SkyWatcher_{m.group(1)}'
    # M12, M14 (Meade) - avoid matching M250 in m250l200 
    m = S(r'(?<![a-z])M\s*(\d{2,3})(?=[^0-9]|$)', text)
    if m:
        val = int(m.group(1))
        if val in (12, 14):
            return f'M{val}'
        elif val in (250, 703, 603):
            return f'M{val}'
    # ACF
    m = S(r'ACF\s*(\d+)', text)
    if m: return f'Meade_ACF{m.group(1)}'
    # Vixen 
    m = S(r'Vixen', text)
    if m: return 'Vixen'
    # Intes
    m = S(r'Intes', text)
    if m: return 'Intes'
    
    # === Refractors: ED, APO, TS ===
    m = S(r'(\d{2,3})\s*ED', text)
    if m: return f'ED{m.group(1)}'
    m = S(r'ED\s*(\d{2,3})', text)
    if m: return f'ED{m.group(1)}'
    # TS80, TS90 etc. (TS-Optics)
    m = S(r'(?<![a-z])TS\s*(\d{2,3})(?=[^0-9]|$)', text)
    if m:
        val = int(m.group(1))
        if val < 200: return f'TS{val}'
    # LU72, Lunette66
    m = S(r'LU\s*(\d+)', text)
    if m: return f'Lunette_{m.group(1)}mm'
    m = S(r'Lunette\s*(\d+)', text)
    if m: return f'Lunette_{m.group(1)}mm'
    
    # === Generic diameter ===
    # "0.6m", "0.51m", ".4m"
    m = S(r'(?:^|[\s])0?\.(\d+)\s*m(?=[^a-z]|$)', text)
    if m:
        frac = m.group(1)
        cm = int(frac) * (10 if len(frac) == 1 else 1)
        return f'T{cm}cm'
    # "60cm", "90 cm" 
    m = S(r'(?:^|[\s])(\d{2,3})\s*cm(?=[^a-z]|$)', text)
    if m: return f'T{m.group(1)}cm'
    # "254mm", "355mm" (reasonable diameter 100-800mm)
    m = S(r'(\d{3})\s*mm', text)
    if m:
        val = int(m.group(1))
        if 100 <= val <= 800:
            return f'T{val}mm'
    # Cassegrain + diameter
    m = S(r'Cassegrain\s*0?\.(\d+)\s*m', text)
    if m:
        frac = m.group(1)
        cm = int(frac) * (10 if len(frac) == 1 else 1)
        return f'Cass_{cm}cm'
    m = S(r'Cassegrain\s*(\d+)', text)
    if m:
        val = int(m.group(1))
        return f'Cass_{val}cm' if val < 100 else f'Cass_{val}mm'
    
    # Inches: "14" or 14" followed by a scope indicator
    m = S(r'(\d+)["\u2033]', text)
    if m:
        val = int(m.group(1))
        if 5 <= val <= 25:
            return f'T{val}inch'
    
    return None


# =============================================================================
# PARSE SPECTROGRAPH
# =============================================================================

def parse_spectrograph(raw):
    """Extract the spectrograph from the fits_bss_inst text."""
    text = normalize(raw)
    
    # === @ Formats ===
    if S(r'Elodie', raw): return 'Elodie'
    if S(r'SOPHIE', raw): return 'SOPHIE'
    if S(r'Feros', raw): return 'FEROS'
    
    # === eShel (very common — many variants) ===
    # eShel, eshel, Eshel, eShel2, eShelV1, eShell, eShel#112, WhoppShel, e-Shel
    if S(r'(?:Whop+shel|e[\s]?shel)', text):
        return 'eShel'
    
    # === LhiresIII / Lhires3 (very common, many variants) ===
    # LHIRES3, LHiRes, Lhires, LhiresIII, LHIRESIII, Lhires III, Lhires lll
    # LIHRES3, LH3, Lh3, LH 2400
    if S(r'l[hi]+res', text):
        return 'LhiresIII'
    if S(r'LH\s*(?:3|III)', text):
        return 'LhiresIII'
    if S(r'(?<![a-z])LH\s*\d{3,4}', text):
        # LH2400, LH1200 etc. — it is LhiresIII with resolution
        return 'LhiresIII'
    # Compact pattern: "L3 150", "L 3"
    if S(r'(?<![a-z])L\s*3(?=[^0-9]|$)', text):
        return 'LhiresIII'
    
    # === Alpy (Shelyak) ===
    m = S(r'Alpy\s*(\d+)?', text)
    if m: return f'Alpy{m.group(1) or "600"}'
    
    # === LISA (Shelyak) ===
    if S(r'(?<![a-z])LISA(?=[^a-z]|$)', text):
        return 'LISA'
    
    # === Star'Ex / StarEx (Shelyak) ===
    if S(r"star[\s'\u2019]*ex", text):
        return 'StarEx'
    
    # === SolEx ===
    if S(r'solex', text):
        return 'SolEx'
    
    # === Star Analyser / SA100 ===
    if S(r'star\s*anal', text):
        return 'StarAnalyser'
    m = S(r'(?<![a-z])SA\s*(?:100|200)', text)
    if m: return 'StarAnalyser'
    
    # === UVEX (Shelyak) ===
    if S(r'uvex', text):
        return 'UVEX'
    
    # === Dados (Baader) ===
    if S(r'dados', text):
        return 'Dados'
    
    # === Specific amateur spectrographs ===
    if S(r'VHIRES', text): return 'VHIRES'
    
    # L200 / Spectra-L200 — do not confuse with "LX200" (Meade telescope)
    m = S(r'(?:Spectra[\s]?)?L[\s]?200(?=[^0-9]|$)', text)
    if m and not S(r'LX\s*200', text):
        return 'Spectra_L200'
    
    if S(r'lowsp|lowres', text): return 'LowSpec'
    if S(r'bespectra', text): return 'BeSpectra'
    if S(r'echelle[\s]?CB', text): return 'Echelle_CB'
    if S(r'dubs[\s]?echelle', text): return 'Dubs_echelle'
    if S(r'SPECTRO[\s]?200', text): return 'Spectro200'
    if S(r'slit[\s]?spec', text): return 'SlitSpec'
    if S(r'(?<![a-z])SGS(?=[^a-z]|$)', text): return 'SGS'
    if S(r'SPOX', text): return 'SPOX'
    if S(r'Rainbow', text): return 'RainbowOptics'
    if S(r'Littrow', text): return 'Littrow'
    if S(r'Bardin', text): return 'Bardin'
    if S(r'JOSIANE', text): return 'JOSIANE'
    if S(r'MEDRES', text): return 'MEDRES'
    if S(r'SARDHIRES', text): return 'SARDHIRES'
    if S(r'Merlette', text): return 'Merlette'
    if S(r'homemade|DIY|custom', text): return 'Homemade'
    
    # === NOU_T / NOU16 / NOU ===
    if S(r'NOU', text):
        return 'NOU_T'
    
    # === Professional ===
    if S(r'HARPS', text): return 'HARPS'
    if S(r'UVES', text): return 'UVES'
    if S(r'X[\s]?Shooter', text): return 'X-Shooter'
    if S(r'AURELIE', text): return 'AURELIE'
    if S(r'MUSICOS', text): return 'MUSICOS'
    if S(r'CARELEC', text): return 'CARELEC'
    if S(r'Veloce', text): return 'Veloce'
    if S(r'(?<![a-z])Neo(?=[^a-z]|$)', text): return 'Neo'
    if S(r'SARG', text): return 'SARG'
    if S(r'GIRAFFE|FLAMES', text): return 'FLAMES_GIRAFFE'
    # Gemini HR
    m = S(r'GEMINI.*?HR', text)
    if m: return 'Gemini_HR'
    
    # === Integrated / named setups ===
    if S(r'DeniSe', raw): return 'DeniSe'
    if S(r'MUSSOL', raw): return 'MUSSOL'
    if S(r'PIERA', raw): return 'PIERA'
    if S(r'ESSAT', text): return 'ESSAT'
    if S(r'(?<![a-z])ESP(?=[^a-z]|$)', text): return 'ESP'
    if S(r'LINX', text): return 'LINX'
    if S(r'REMOT', text): return 'REMOT'
    
    # === IUE (satellite) ===
    if S(r'IUE[\s]?SWP', text): return 'IUE_SWP'
    if S(r'IUE[\s]?LWP', text): return 'IUE_LWP'
    if S(r'IUE[\s]?LWR', text): return 'IUE_LWR'
    
    # === Generic echelle ===
    if S(r'echelle', text) and not S(r'eshel|CB|dubs', text):
        return 'Echelle'
    
    # === Compact pattern: V10S2400, G15S2400, T10S300 ===
    m = S(r'[A-Z]\d+S(\d+)$', raw.strip())
    if m: return f'Compact_R{m.group(1)}'
    
    # === Grating only at end of string: "+ 1200", "/ 1200" ===
    if S(r'[\s](\d{3,4})\s*$', text):
        return 'Unknown_grating'
    
    return None


# =============================================================================
# PARSE DETECTOR
# =============================================================================

def parse_detector(raw):
    """Extract the detector/camera from the fits_bss_inst text."""
    text = normalize(raw)
    
    # === Atik / ATK (very common) ===
    m = S(r'(?:ATIK|ATK)\s*(\d+)', text)
    if m: return f'Atik{m.group(1)}'
    # "AtikL" without numero → Atik-Line (modele without numero)
    m = S(r'Atik\s*L(?=[^0-9]|$)', text)
    if m: return 'AtikL'
    
    # === ZWO / ASI ===
    m = S(r'(?:ZWO\s*)?ASI\s*(\d+)', text)
    if m: return f'ASI{m.group(1)}'
    
    # === QSI ===
    m = S(r'QSI\s*(\d+)', text)
    if m: return f'QSI{m.group(1)}'
    
    # === QHY ===
    m = S(r'QHY\s*(?:IMG)?\s*(\d+)', text)
    if m: return f'QHY{m.group(1)}'
    
    # === SBIG / ST ===
    m = S(r'SBIG\s*(?:ST\s*)?(\d+)', text)
    if m: return f'SBIG_ST{m.group(1)}'
    # ST pattern — attention ne pas matcher in ESSAT, eShel contexts
    m = S(r'(?<![a-z])ST\s*(\d{1,5})(?:XME|ME|M)?(?=[^0-9]|$)', text)
    if m:
        # Verify this is not a false positive
        start = m.start()
        prefix = text[max(0, start-4):start].strip()
        if prefix.upper() not in ('ESSA', 'ES', 'ESS'):
            return f'SBIG_ST{m.group(1)}'
    
    # === Audine ===
    m = S(r'Audine\s*(?:KAF\s*)?(\d+)?', text)
    if m: return f'Audine{m.group(1) or ""}'
    
    # === KAF (CCD chip) ===
    m = S(r'KAF\s*(\d+)', text)
    if m: return f'KAF{m.group(1)}'
    
    # === Trius / SXVR ===
    m = S(r'(?:TRIUS|Trius)\s*(?:SX)?\s*[A-Z]?\s*(\d+)', text)
    if m: return f'Trius_{m.group(1)}'
    m = S(r'SXVR\s*H\s*(\d+)', text)
    if m: return f'SXVR_H{m.group(1)}'
    m = S(r'SXV\s*[MH]\s*(\d+)', text)
    if m: return f'SXVR_{m.group(1)}'
    m = S(r'(?<![a-z])SX\s*(\d+)', text)
    if m: return f'SX_{m.group(1)}'
    
    # === FLI / ML ===
    m = S(r'FLI\s*(?:ML)?\s*(\d+)', text)
    if m: return f'FLI_ML{m.group(1)}'
    m = S(r'(?<![a-z])ML\s*(\d+)', text)
    if m: return f'FLI_ML{m.group(1)}'
    
    # === CCD generic ===
    m = S(r'CCD\s*(\d+)', text)
    if m: return f'CCD{m.group(1)}'
    
    # === Meade DSI ===
    if S(r'DSI', text): return 'Meade_DSI'
    
    # === Nova ===
    m = S(r'Nova\s*(\d+)', text)
    if m: return f'Nova{m.group(1)}'
    
    # === MX ===
    m = S(r'(?<![a-z])MX\s*(\d+)', text)
    if m: return f'MX{m.group(1)}'
    
    # === Canon / DSLR ===
    m = S(r'Canon\s*(\d+)', text)
    if m: return f'Canon{m.group(1)}'
    if S(r'Nikon', text): return 'Nikon'
    
    # === HISIS ===
    if S(r'HISIS', text): return 'HISIS'
    
    # === TH7852 ===
    if S(r'TH\s*7852', text): return 'TH7852'
    
    # === Andor / iKon ===
    if S(r'iKon', text): return 'Andor_iKon'
    if S(r'Andor', text): return 'Andor'
    
    # === Moravian ===
    if S(r'Moravian', text): return 'Moravian'
    
    # === LINX (detector in "16 REMOT LINX 460EX") ===
    # Note: LINX is a spectrograph, not a detector - do not match here
    
    # === ARES ===
    if S(r'ARES', text): return 'ARES'
    
    # === ILX511 ===
    if S(r'ILX\s*511', text): return 'ILX511'
    
    # === STL, STF (SBIG advanced) ===
    m = S(r'ST[LF]\s*(\d+)', text)
    if m: return f'SBIG_STL{m.group(1)}'
    
    # === Implicit detector: "460" alone (probably Atik460 in BeSS) ===
    # No — too risky as an assumption
    
    return None


# =============================================================================
# CONFIDENCE & SETUP NAME
# =============================================================================

# Professional spectrographs → known implicit telescope
PROFESSIONAL_SETUPS = {
    'SOPHIE': 'OHP_T193',
    'Elodie': 'OHP_T193',
    'FEROS': None,  # ESO 1.52 or 2.2 — already in @ESO
    'HARPS': 'ESO_3.6m',
    'UVES': 'ESO_VLT',
    'X-Shooter': 'ESO_VLT',
    'AURELIE': 'OHP_T152',
    'CARELEC': 'OHP_T152',
    'MUSICOS': 'PicDuMidi_TBL',
    'Veloce': 'AAO_3.9m',
    'SARG': 'TNG_3.6m',
    'FLAMES_GIRAFFE': 'ESO_VLT',
    'IUE_SWP': 'IUE',
    'IUE_LWP': 'IUE',
    'IUE_LWR': 'IUE',
}

# Integrated setups (proper names, not decomposable)
INTEGRATED_SETUPS = {'DeniSe', 'MUSSOL', 'PIERA', 'NOU_T', 'LINX', 'REMOT', 'ESP', 'ESSAT', 'Spectro200'}


def enrich_professional(telescope, spectrograph):
    """Enrich the telescope for professional spectrographs."""
    if telescope is None and spectrograph in PROFESSIONAL_SETUPS:
        return PROFESSIONAL_SETUPS[spectrograph]
    return telescope


def determine_confidence(telescope, spectrograph, detector, raw):
    """Determine the confidence level."""
    components = sum(1 for x in (telescope, spectrograph, detector) if x is not None)
    
    if components >= 2:
        return 'high'
    elif components == 1:
        # Integrated or professional setups = OK with 1 component
        if spectrograph in INTEGRATED_SETUPS or spectrograph in PROFESSIONAL_SETUPS:
            return 'high'
        # Telescope alone (e.g. SC14, SC25) is medium
        return 'medium'
    else:
        return 'low'


def build_setup_name(telescope, spectrograph, detector):
    """Build the canonical setup name."""
    parts = []
    if telescope:
        parts.append(telescope)
    if spectrograph:
        parts.append(spectrograph)
    if detector:
        parts.append(detector)
    return '+'.join(parts) if parts else None


# =============================================================================
# MAIN
# =============================================================================

def main():
    db_path = Path(__file__).parent.parent.parent / 'harvest' / 'bess_data.duckdb'
    output_path = Path(__file__).parent / 'instrument_lookup.json'
    
    print(f"Connecting to {db_path}")
    conn = duckdb.connect(str(db_path), read_only=True)
    
    results = conn.execute('''
        SELECT fits_bss_inst, COUNT(*) as cnt 
        FROM spectra_votable 
        WHERE fits_bss_inst IS NOT NULL
        GROUP BY fits_bss_inst
        ORDER BY cnt DESC
    ''').fetchall()
    
    conn.close()
    
    total_spectra = sum(r[1] for r in results)
    print(f"\nUnique setups: {len(results)}")
    print(f"Total spectra: {total_spectra:,}")
    
    lookup = {}
    stats = Counter()
    
    for raw, count in results:
        telescope = parse_telescope(raw)
        spectrograph = parse_spectrograph(raw)
        detector = parse_detector(raw)
        
        # Professional enrichment
        telescope = enrich_professional(telescope, spectrograph)
        
        confidence = determine_confidence(telescope, spectrograph, detector, raw)
        setup_name = build_setup_name(telescope, spectrograph, detector)
        
        lookup[raw] = {
            'instrument_setup': setup_name,
            'telescope': telescope,
            'spectrograph': spectrograph,
            'detector': detector,
            'confidence': confidence,
            'spectra_count': count,
        }
        
        stats[confidence] += 1
    
    # Write JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(lookup, f, indent=2, ensure_ascii=False)
    
    print(f"\nLookup table written: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024:.0f} KB")
    
    # ===== STATISTICS =====
    print("\n" + "=" * 70)
    print("PARSING STATISTICS")
    print("=" * 70)
    
    # Confidence
    print(f"\n--- Confidence ---")
    for level in ('high', 'medium', 'low'):
        n = stats[level]
        spectra = sum(v['spectra_count'] for v in lookup.values() if v['confidence'] == level)
        print(f"  {level:<8}: {n:>4} setups ({n/len(results)*100:.1f}%)   {spectra:>8,} spectra ({spectra/total_spectra*100:.1f}%)")
    
    # Detection by component
    tel_ok = sum(1 for v in lookup.values() if v['telescope'])
    spec_ok = sum(1 for v in lookup.values() if v['spectrograph'])
    det_ok = sum(1 for v in lookup.values() if v['detector'])
    tel_sp = sum(v['spectra_count'] for v in lookup.values() if v['telescope'])
    spec_sp = sum(v['spectra_count'] for v in lookup.values() if v['spectrograph'])
    det_sp = sum(v['spectra_count'] for v in lookup.values() if v['detector'])
    
    print(f"\n--- Detection by component ---")
    print(f"  {'Component':<15} {'Setups':>8} {'%':>6}  {'Spectra':>10} {'%':>6}")
    print(f"  {'Telescope':<15} {tel_ok:>8} {tel_ok/len(results)*100:>5.1f}%  {tel_sp:>10,} {tel_sp/total_spectra*100:>5.1f}%")
    print(f"  {'Spectrograph':<15} {spec_ok:>8} {spec_ok/len(results)*100:>5.1f}%  {spec_sp:>10,} {spec_sp/total_spectra*100:>5.1f}%")
    print(f"  {'Detector':<15} {det_ok:>8} {det_ok/len(results)*100:>5.1f}%  {det_sp:>10,} {det_sp/total_spectra*100:>5.1f}%")
    
    # Unique values
    telescopes = Counter(v['telescope'] for v in lookup.values() if v['telescope'])
    spectrographs = Counter(v['spectrograph'] for v in lookup.values() if v['spectrograph'])
    detectors = Counter(v['detector'] for v in lookup.values() if v['detector'])
    
    print(f"\n--- Unique values ---")
    print(f"  Telescopes:     {len(telescopes)}")
    print(f"  Spectrographs:  {len(spectrographs)}")
    print(f"  Detectors:      {len(detectors)}")
    
    # Top 20 spectrographs by spectra
    spec_by_sp = Counter()
    for v in lookup.values():
        if v['spectrograph']:
            spec_by_sp[v['spectrograph']] += v['spectra_count']
    print(f"\n--- Top 20 spectrographs (by spectra) ---")
    for s, c in spec_by_sp.most_common(20):
        print(f"  {s:<20} {c:>8,} spectra")
    
    # Top 20 telescopes by spectra
    tel_by_sp = Counter()
    for v in lookup.values():
        if v['telescope']:
            tel_by_sp[v['telescope']] += v['spectra_count']
    print(f"\n--- Top 20 telescopes (by spectra) ---")
    for t, c in tel_by_sp.most_common(20):
        print(f"  {t:<20} {c:>8,} spectra")
    
    # Top 20 detectors by spectra
    det_by_sp = Counter()
    for v in lookup.values():
        if v['detector']:
            det_by_sp[v['detector']] += v['spectra_count']
    print(f"\n--- Top 20 detectors (by spectra) ---")
    for d, c in det_by_sp.most_common(20):
        print(f"  {d:<20} {c:>8,} spectra")
    
    # LOW cases
    low = [(r, v) for r, v in lookup.items() if v['confidence'] == 'low']
    low.sort(key=lambda x: -x[1]['spectra_count'])
    print(f"\n--- LOW confidence cases ({len(low)} setups, {sum(v['spectra_count'] for _,v in low)} spectra) ---")
    for r, v in low:
        print(f"  {v['spectra_count']:>5} | tel={str(v['telescope']):<20} spec={str(v['spectrograph']):<15} det={str(v['detector']):<15} | {r}")
    
    # MEDIUM cases (top 30)
    med = [(r, v) for r, v in lookup.items() if v['confidence'] == 'medium']
    med.sort(key=lambda x: -x[1]['spectra_count'])
    print(f"\n--- MEDIUM confidence cases (top 30 / {len(med)} total, {sum(v['spectra_count'] for _,v in med)} spectra) ---")
    for r, v in med[:30]:
        print(f"  {v['spectra_count']:>5} | tel={str(v['telescope']):<20} spec={str(v['spectrograph']):<15} det={str(v['detector']):<15} | {r}")


if __name__ == '__main__':
    main()
