package com.wolof.dataset;

import com.wolof.model.WolofPhrase;
import org.springframework.stereotype.Component;

import java.util.*;

/**
 * ============================================================
 *  GÉNÉRATEUR DE DATASET WOLOF
 * ============================================================
 *
 * Génère ~5000 phrases wolof synthétiques à partir du vocabulaire
 * fourni dans le sujet du mémoire.
 *
 * Structure d'une phrase générée :
 *   [Sujet] + [Verbe] + [Complément] + [Adverbe (optionnel)]
 *
 * Exemple :
 *   Latin    : "Maa ngi dem ci marché bi tey"
 *   Officiel : "Maa ngi dem ci màrse bi tey"
 *   Ajami    : "ما نقي ديم سي مارشي بي تاي"
 * ============================================================
 */
@Component
public class WolofDatasetGenerator {

    // -------------------------------------------------------
    // VOCABULAIRE SOURCE (latin courant)
    // -------------------------------------------------------

    private static final String[] SUJETS = {
        "Maa ngi", "Yaa ngi", "Mu ngi", "Ñu ngi", "Góor gi", "Jigeen ji",
        "Xale bi", "Baay bi", "Ndey ji", "Mame bi", "Nit ñi", "Doom ji",
        "Jàngalekat bi", "Liggéeykat bi", "Damaa", "Bëgg naa", "Jënd naa",
        "Jaay naa", "Lekk naa", "Naan naa", "Nelaw naa", "Togg naa",
        "Wax naa", "Déglu naa", "Gis naa", "Dakar", "Touba", "Thiès",
        "Saint-Louis", "Ziguinchor", "Kaolack"
    };

    private static final String[] VERBES = {
        "dem ci", "jóge ci", "toog ci", "nelaw ci", "togg ci", "jàng ci",
        "jàngal ci", "liggéey ci", "wax ci", "déglu ci", "gis ci", "jël ci",
        "dëkk ci", "sant ci", "ñëw ci", "jëli", "indi", "sanni", "tëj",
        "ubbi", "daw ci", "dox ci"
    };

    private static final String[] COMPLEMENTS = {
        "marché bi", "école bi", "bureau bi", "garab gi", "dex gi", "tool bi",
        "kër gi", "mosquée bi", "église bi", "tali bi", "mbëd mi", "all bi",
        "genn bi", "bëj-gànnaar bi", "soow bi", "ceeb bi", "mburu mi",
        "ndox mi", "meew mi", "nag wi", "xar mi", "fas wi", "gaal gi",
        "auto bi", "woto bi", "téere bi", "xaliss bi", "jëkkër ji", "baatu bi"
    };

    private static final String[] ADVERBES = {
        "bu baax", "bu rëy", "bu tàng", "bu gaaw", "bu yàgg", "tey", "suba",
        "kér", "démb", "léegi", "léegi-léegi", "bu sedd", "bu ñuul",
        "bu neex", "ba lool", ""  // "" = pas d'adverbe (phrase sans adverbe)
    };

    // -------------------------------------------------------
    // TABLE DE CORRESPONDANCE LATIN → OFFICIEL CLAD
    // -------------------------------------------------------

    /**
     * Règles de correspondance entre écriture latine courante et CLAD officiel.
     * LinkedHashMap pour respecter l'ordre (les règles plus longues d'abord).
     */
    private static final Map<String, String> LATIN_TO_CLAD = new LinkedHashMap<>();

    static {
        // Règles multi-caractères (à appliquer en premier)
        LATIN_TO_CLAD.put("léegi-léegi", "léegi-léegi"); // inchangé
        LATIN_TO_CLAD.put("bëj-gànnaar", "bëj-gànnaar"); // inchangé
        LATIN_TO_CLAD.put("marché",     "màrse");
        LATIN_TO_CLAD.put("école",      "ekol");
        LATIN_TO_CLAD.put("bureau",     "biiro");
        LATIN_TO_CLAD.put("mosquée",    "jaami");
        LATIN_TO_CLAD.put("église",     "egliz");
        LATIN_TO_CLAD.put("auto",       "ooto");
        LATIN_TO_CLAD.put("Saint-Louis","Ndar");
        LATIN_TO_CLAD.put("Ziguinchor", "Siggincoor");
        LATIN_TO_CLAD.put("jàngal",     "jàngal");   // inchangé
        LATIN_TO_CLAD.put("liggéey",    "liggéey");  // inchangé
        LATIN_TO_CLAD.put("jàng",       "jàng");     // inchangé
        // Règles simples
        LATIN_TO_CLAD.put("ñëw",        "ñëw");
        LATIN_TO_CLAD.put("dëkk",       "dëkk");
        LATIN_TO_CLAD.put("jënd",       "jënd");
        LATIN_TO_CLAD.put("jëkkër",     "jëkkër");
        LATIN_TO_CLAD.put("mburu",      "mburu");
        LATIN_TO_CLAD.put("ceeb",       "ceebu");
        LATIN_TO_CLAD.put("xaliss",     "xaalis");
        LATIN_TO_CLAD.put("téere",      "tëyel");
        LATIN_TO_CLAD.put("woto",       "woto");
        LATIN_TO_CLAD.put("garab",      "garab");
        LATIN_TO_CLAD.put("gànnaar",    "gànnaar");
        LATIN_TO_CLAD.put("Thiès",      "Cees");
        LATIN_TO_CLAD.put("Kaolack",    "Kawlax");
    }

    // -------------------------------------------------------
    // TABLE DE CORRESPONDANCE LATIN → AJAMI
    // -------------------------------------------------------

    /** Dictionnaire de translittération vers l'écriture Ajami (arabe wolof) */
    private static final Map<String, String> DICO_AJAMI = new LinkedHashMap<>();

    static {
        DICO_AJAMI.put("Maa", "ما");      DICO_AJAMI.put("ngi", "نقي");
        DICO_AJAMI.put("Yaa", "يا");      DICO_AJAMI.put("Mu", "مو");
        DICO_AJAMI.put("Ñu", "نو");       DICO_AJAMI.put("gi", "قي");
        DICO_AJAMI.put("ji", "جي");       DICO_AJAMI.put("bi", "بي");
        DICO_AJAMI.put("Góor", "قور");    DICO_AJAMI.put("Jigeen", "جيقين");
        DICO_AJAMI.put("Xale", "خال");    DICO_AJAMI.put("Baay", "باي");
        DICO_AJAMI.put("Ndey", "نداي");   DICO_AJAMI.put("Mame", "مام");
        DICO_AJAMI.put("Nit", "نيت");     DICO_AJAMI.put("ñi", "ني");
        DICO_AJAMI.put("Doom", "دوم");    DICO_AJAMI.put("Jàngalekat", "جاقاليكات");
        DICO_AJAMI.put("Liggéeykat", "ليقيكات"); DICO_AJAMI.put("Damaa", "داما");
        DICO_AJAMI.put("Bëgg", "بكّ");    DICO_AJAMI.put("naa", "نا");
        DICO_AJAMI.put("Jënd", "جند");    DICO_AJAMI.put("Jaay", "جاي");
        DICO_AJAMI.put("Lekk", "ليك");    DICO_AJAMI.put("Naan", "نان");
        DICO_AJAMI.put("Nelaw", "نيلاو"); DICO_AJAMI.put("Togg", "توق");
        DICO_AJAMI.put("Wax", "واكس");    DICO_AJAMI.put("Déglu", "دقلو");
        DICO_AJAMI.put("Gis", "قيس");     DICO_AJAMI.put("Dakar", "داكار");
        DICO_AJAMI.put("Touba", "توبا");  DICO_AJAMI.put("Thiès", "تياس");
        DICO_AJAMI.put("Saint-Louis", "سانت-لوي"); DICO_AJAMI.put("Ziguinchor", "زيقينشور");
        DICO_AJAMI.put("Kaolack", "كاولاك"); DICO_AJAMI.put("dem", "ديم");
        DICO_AJAMI.put("ci", "سي");       DICO_AJAMI.put("jóge", "جوق");
        DICO_AJAMI.put("toog", "توق");    DICO_AJAMI.put("nelaw", "نيلاو");
        DICO_AJAMI.put("togg", "توق");    DICO_AJAMI.put("jàng", "جاق");
        DICO_AJAMI.put("jàngal", "جاقال"); DICO_AJAMI.put("liggéey", "ليقي");
        DICO_AJAMI.put("wax", "واكس");    DICO_AJAMI.put("déglu", "دقلو");
        DICO_AJAMI.put("gis", "قيس");     DICO_AJAMI.put("jël", "جل");
        DICO_AJAMI.put("dëkk", "دكّ");   DICO_AJAMI.put("sant", "سانت");
        DICO_AJAMI.put("ñëw", "نو");      DICO_AJAMI.put("jëli", "جلي");
        DICO_AJAMI.put("indi", "يندي");   DICO_AJAMI.put("sanni", "صاني");
        DICO_AJAMI.put("tëj", "تج");      DICO_AJAMI.put("ubbi", "اوبي");
        DICO_AJAMI.put("daw", "داو");     DICO_AJAMI.put("dox", "دوح");
        DICO_AJAMI.put("marché", "مارشي"); DICO_AJAMI.put("école", "ايكول");
        DICO_AJAMI.put("bureau", "بيرو"); DICO_AJAMI.put("garab", "قراب");
        DICO_AJAMI.put("dex", "ديخ");     DICO_AJAMI.put("tool", "تول");
        DICO_AJAMI.put("kër", "كر");      DICO_AJAMI.put("mosquée", "مسجد");
        DICO_AJAMI.put("église", "ايقليز"); DICO_AJAMI.put("tali", "طالي");
        DICO_AJAMI.put("mbëd", "مبد");    DICO_AJAMI.put("all", "عال");
        DICO_AJAMI.put("genn", "قين");    DICO_AJAMI.put("bëj-gànnaar", "بج-قانار");
        DICO_AJAMI.put("soow", "صو");     DICO_AJAMI.put("ceeb", "سيب");
        DICO_AJAMI.put("mburu", "مبورو"); DICO_AJAMI.put("ndox", "ندوح");
        DICO_AJAMI.put("meew", "ميو");    DICO_AJAMI.put("nag", "ناق");
        DICO_AJAMI.put("xar", "خار");     DICO_AJAMI.put("fas", "فاس");
        DICO_AJAMI.put("gaal", "قال");    DICO_AJAMI.put("auto", "اوطو");
        DICO_AJAMI.put("woto", "ووطو");   DICO_AJAMI.put("téere", "طير");
        DICO_AJAMI.put("xaliss", "خاليس"); DICO_AJAMI.put("jëkkër", "جكر");
        DICO_AJAMI.put("baatu", "باتو");  DICO_AJAMI.put("bu", "بو");
        DICO_AJAMI.put("baax", "باه");    DICO_AJAMI.put("rëy", "ري");
        DICO_AJAMI.put("tàng", "طاق");    DICO_AJAMI.put("gaaw", "قاو");
        DICO_AJAMI.put("yàgg", "ياق");    DICO_AJAMI.put("tey", "تاي");
        DICO_AJAMI.put("suba", "صوبا");   DICO_AJAMI.put("kér", "كير");
        DICO_AJAMI.put("démb", "ديمب");   DICO_AJAMI.put("léegi", "ليقي");
        DICO_AJAMI.put("léegi-léegi", "ليقي-ليقي");
        DICO_AJAMI.put("sedd", "سيدّ");   DICO_AJAMI.put("ñuul", "نول");
        DICO_AJAMI.put("neex", "نيح");    DICO_AJAMI.put("ba", "با");
        DICO_AJAMI.put("lool", "لول");    DICO_AJAMI.put("wi", "وي");
        DICO_AJAMI.put("mi", "مي");
    }

    // -------------------------------------------------------
    // GÉNÉRATION DU DATASET
    // -------------------------------------------------------

    /**
     * Génère le dataset complet (~5000 phrases).
     *
     * @return Liste de paires (latin, officiel, ajami)
     */
    public List<WolofPhrase> generateDataset() {
        Set<String> seen = new HashSet<>();   // Pour éviter les doublons
        List<WolofPhrase> dataset = new ArrayList<>();
        Random rnd = new Random(42);           // Seed fixe pour reproductibilité

        for (String sujet : SUJETS) {
            for (String verbe : VERBES) {
                for (String complement : COMPLEMENTS) {
                    // Générer avec ET sans adverbe
                    for (String adverbe : ADVERBES) {
                        String latin = buildLatinPhrase(sujet, verbe, complement, adverbe);

                        // Éviter les doublons
                        if (seen.contains(latin)) continue;
                        seen.add(latin);

                        String officiel = toOfficiel(latin);
                        String ajami    = toAjami(latin);

                        dataset.add(new WolofPhrase(latin, officiel, ajami));

                        // Arrêter à 5000
                        if (dataset.size() >= 5000) return dataset;
                    }
                }
            }
        }
        return dataset;
    }

    /**
     * Assemble une phrase en latin courant.
     * Ex : "Maa ngi" + "dem ci" + "marché bi" + "tey" → "Maa ngi dem ci marché bi tey"
     */
    private String buildLatinPhrase(String sujet, String verbe, String complement, String adverbe) {
        StringBuilder sb = new StringBuilder();
        sb.append(sujet).append(" ").append(verbe).append(" ").append(complement);
        if (!adverbe.isEmpty()) {
            sb.append(" ").append(adverbe);
        }
        return sb.toString().trim();
    }

    // -------------------------------------------------------
    // TRANSLITTÉRATION PAR RÈGLES
    // -------------------------------------------------------

    /**
     * Convertit un texte latin en orthographe officielle CLAD.
     * Applique les règles définies dans LATIN_TO_CLAD mot par mot.
     */
    public String toOfficiel(String latinText) {
        String result = latinText;
        // Appliquer chaque règle de remplacement
        for (Map.Entry<String, String> rule : LATIN_TO_CLAD.entrySet()) {
            result = result.replace(rule.getKey(), rule.getValue());
        }
        return result;
    }

    /**
     * Convertit un texte officiel en latin courant (inverse de toOfficiel).
     */
    public String toLatinFromOfficiel(String officielText) {
        String result = officielText;
        // Parcourir les règles en sens inverse
        List<Map.Entry<String, String>> entries = new ArrayList<>(LATIN_TO_CLAD.entrySet());
        Collections.reverse(entries);
        for (Map.Entry<String, String> rule : entries) {
            result = result.replace(rule.getValue(), rule.getKey());
        }
        return result;
    }

    /**
     * Convertit un texte latin en écriture Ajami (arabe wolof).
     * Traite le texte mot par mot.
     */
    public String toAjami(String latinText) {
        String[] words = latinText.split("\\s+");
        StringBuilder ajami = new StringBuilder();
        for (String word : words) {
            // Chercher une correspondance dans le dictionnaire
            String translated = DICO_AJAMI.getOrDefault(word,
                                DICO_AJAMI.getOrDefault(word.toLowerCase(), word));
            ajami.append(translated).append(" ");
        }
        return ajami.toString().trim();
    }

    /**
     * Calcule un score d'attention simulé pour un token.
     * Un token long ou avec une correspondance exacte obtient un score élevé.
     * Ce score sera remplacé par les vraies valeurs d'attention du modèle Seq2Seq.
     *
     * @param srcToken token source
     * @param tgtToken token cible
     * @return score entre 0.0 et 1.0
     */
    public double computeAttentionScore(String srcToken, String tgtToken) {
        if (srcToken.equals(tgtToken)) return 0.9;          // Tokens identiques
        if (LATIN_TO_CLAD.containsKey(srcToken)) return 0.95; // Règle exacte trouvée
        if (DICO_AJAMI.containsKey(srcToken)) return 0.85;  // Dans le dico ajami
        // Score basé sur la longueur relative
        double ratio = (double) Math.min(srcToken.length(), tgtToken.length())
                     / Math.max(srcToken.length(), tgtToken.length());
        return 0.3 + ratio * 0.5;
    }
}
