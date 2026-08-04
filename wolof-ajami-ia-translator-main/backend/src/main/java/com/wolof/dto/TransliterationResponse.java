package com.wolof.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * Objet retourné à Angular après translittération.
 *
 * Exemple JSON renvoyé :
 * {
 *   "input": "Maa ngi dem ci marché bi",
 *   "output": "Maa ngi dem ci màrse bi",
 *   "direction": "lat2off",
 *   "tokensSrc": ["Maa", "ngi", "dem", "ci", "marché", "bi"],
 *   "tokensTgt": ["Maa", "ngi", "dem", "ci", "màrse", "bi"],
 *   "attentionScores": [0.9, 0.85, 0.7, 0.6, 0.95, 0.8],
 *   "processingTimeMs": 12
 * }
 */
@Data
@Builder                // Permet la syntaxe : TransliterationResponse.builder().output(...).build()
@NoArgsConstructor      // Constructeur sans argument (requis par Jackson pour la sérialisation JSON)
@AllArgsConstructor     // Constructeur avec tous les arguments
public class TransliterationResponse {

    /** Texte original envoyé par l'utilisateur */
    private String input;

    /** Texte translittéré résultant */
    private String output;

    /** Direction utilisée */
    private String direction;

    /** Tokens du texte source (un token = un mot) */
    private List<String> tokensSrc;

    /** Tokens du texte cible correspondants */
    private List<String> tokensTgt;

    /**
     * Scores d'attention pour chaque paire de tokens (entre 0 et 1).
     * Simule le mécanisme d'attention du modèle Seq2Seq.
     * Plus le score est élevé, plus la correspondance est forte.
     */
    private List<Double> attentionScores;

    /** Temps de traitement en millisecondes */
    private long processingTimeMs;
}
