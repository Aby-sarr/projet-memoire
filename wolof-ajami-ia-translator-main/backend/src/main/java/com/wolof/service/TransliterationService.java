package com.wolof.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.wolof.dataset.WolofDatasetGenerator;
import com.wolof.dto.TransliterationRequest;
import com.wolof.dto.TransliterationResponse;
import com.wolof.model.WolofPhrase;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * ============================================================
 * SERVICE DE TRANSLITTÉRATION
 * ============================================================
 *
 * Ce service assure la communication entre :
 *
 * Angular
 *    ↓
 * Spring Boot
 *    ↓
 * FastAPI
 *    ↓
 * Modèle Seq2Seq avec attention
 *
 * Directions prises en charge :
 *
 * - lat2off   : Latin courant → Orthographe officielle CLAD
 * - off2lat   : Orthographe officielle CLAD → Latin courant
 * - lat2ajami : Latin → Ajami
 * - ajami2lat : Ajami → Latin
 *
 * Les directions lat2ajami et ajami2lat utilisent
 * le modèle IA Python/FastAPI.
 * ============================================================
 */
@Service
@Slf4j
public class TransliterationService {

    private final WolofDatasetGenerator generator;
    private final RestClient restClient;
    private final ObjectMapper objectMapper;

    /**
     * ============================================================
     * ADRESSE DE L'API FASTAPI
     * ============================================================
     *
     * FastAPI fonctionne sur :
     *
     * http://127.0.0.1:8000
     */
    private static final String AI_API_URL =
            "http://127.0.0.1:8000";

    /**
     * ============================================================
     * CONSTRUCTEUR
     * ============================================================
     */
    public TransliterationService(
            WolofDatasetGenerator generator) {

        this.generator = generator;

        this.restClient = RestClient.builder()
                .baseUrl(AI_API_URL)
                .build();

        this.objectMapper = new ObjectMapper();
    }

    /**
     * ============================================================
     * POINT D'ENTRÉE PRINCIPAL
     * ============================================================
     */
    public TransliterationResponse transliterate(
            TransliterationRequest request) {

        long startTime =
                System.currentTimeMillis();

        String text =
                request.getText();

        String direction =
                request.getDirection();

        log.info(
                "Translittération : '{}' [{}]",
                text,
                direction
        );

        /*
         * --------------------------------------------------------
         * 1. TOKENISATION DU TEXTE SOURCE
         * --------------------------------------------------------
         */
        List<String> srcTokens =
                tokenize(text);

        /*
         * --------------------------------------------------------
         * 2. TRANSLITTÉRATION SELON LA DIRECTION
         * --------------------------------------------------------
         */
        String outputText;

        switch (direction) {

            /*
             * ====================================================
             * LATIN → AJAMI
             * ====================================================
             */
            case "lat2ajami":

                outputText =
                        callAiModel(
                                text,
                                "lat2ajami"
                        );

                break;

            /*
             * ====================================================
             * AJAMI → LATIN
             * ====================================================
             */
            case "ajami2lat":

                outputText =
                        callAiModel(
                                text,
                                "ajami2lat"
                        );

                break;

            /*
             * ====================================================
             * LATIN → ORTHOGRAPHE OFFICIELLE
             * ====================================================
             */
            case "lat2off":

                outputText =
                        generator.toOfficiel(text);

                break;

            /*
             * ====================================================
             * ORTHOGRAPHE OFFICIELLE → LATIN
             * ====================================================
             */
            case "off2lat":

                outputText =
                        generator.toLatinFromOfficiel(text);

                break;

            default:

                throw new IllegalArgumentException(
                        "Direction invalide : "
                                + direction
                );
        }

        /*
         * --------------------------------------------------------
         * 3. TOKENISATION DU RÉSULTAT
         * --------------------------------------------------------
         */
        List<String> tgtTokens =
                tokenize(outputText);

        /*
         * --------------------------------------------------------
         * 4. CALCUL DES SCORES D'ATTENTION
         * --------------------------------------------------------
         *
         * Les vrais poids de Bahdanau sont calculés
         * dans le modèle Python.
         *
         * FastAPI ne les expose actuellement pas
         * directement dans sa réponse.
         *
         * On conserve donc le calcul uniforme existant
         * utilisé par l'application Spring Boot.
         */
        List<Double> attentionScores =
                computeAttentionScores(
                        srcTokens,
                        tgtTokens
                );

        long processingTime =
                System.currentTimeMillis()
                        - startTime;

        log.info(
                "Résultat : '{}' en {} ms",
                outputText,
                processingTime
        );

        /*
         * --------------------------------------------------------
         * 5. RÉPONSE ENVOYÉE À ANGULAR
         * --------------------------------------------------------
         */
        return TransliterationResponse.builder()
                .input(text)
                .output(outputText)
                .direction(direction)
                .tokensSrc(srcTokens)
                .tokensTgt(tgtTokens)
                .attentionScores(attentionScores)
                .processingTimeMs(processingTime)
                .build();
    }

    /**
     * ============================================================
     * APPEL DU MODÈLE IA PYTHON / FASTAPI
     * ============================================================
     *
     * FastAPI possède deux endpoints :
     *
     * Latin → Ajami :
     *
     * POST http://127.0.0.1:8000/predict
     *
     * Ajami → Latin :
     *
     * POST http://127.0.0.1:8000/predict_reverse
     *
     * Le corps envoyé est uniquement :
     *
     * {
     *     "text": "xam"
     * }
     *
     * ou :
     *
     * {
     *     "text": "خام"
     * }
     *
     * FastAPI retourne :
     *
     * {
     *     "input": "...",
     *     "output": "...",
     *     "direction": "..."
     * }
     */
    private String callAiModel(
            String text,
            String direction) {

        try {

            /*
             * ----------------------------------------------------
             * CHOIX DE L'ENDPOINT
             * ----------------------------------------------------
             */
            String endpoint;

            if ("lat2ajami".equals(direction)) {

                /*
                 * Latin → Ajami
                 */
                endpoint = "/predict";

            } else if ("ajami2lat".equals(direction)) {

                /*
                 * Ajami → Latin
                 */
                endpoint = "/predict_reverse";

            } else {

                throw new IllegalArgumentException(
                        "Direction IA invalide : "
                                + direction
                );
            }

            log.info(
                    "Appel FastAPI : {}{}",
                    AI_API_URL,
                    endpoint
            );

            /*
             * ----------------------------------------------------
             * REQUÊTE ENVOYÉE À FASTAPI
             * ----------------------------------------------------
             *
             * IMPORTANT :
             *
             * L'API FastAPI actuelle attend uniquement :
             *
             * {
             *     "text": "..."
             * }
             *
             * Il ne faut PAS envoyer "direction".
             */
            String response =
                    restClient.post()
                            .uri(endpoint)
                            .contentType(
                                    MediaType.APPLICATION_JSON
                            )
                            .body(
                                    new PredictionRequest(
                                            text
                                    )
                            )
                            .retrieve()
                            .body(String.class);

            /*
             * ----------------------------------------------------
             * VÉRIFICATION DE LA RÉPONSE
             * ----------------------------------------------------
             */
            if (response == null ||
                    response.isBlank()) {

                throw new RuntimeException(
                        "Réponse vide de FastAPI"
                );
            }

            log.info(
                    "Réponse FastAPI : {}",
                    response
            );

            /*
             * ----------------------------------------------------
             * LECTURE DU JSON
             * ----------------------------------------------------
             */
            JsonNode json =
                    objectMapper.readTree(response);

            /*
             * ----------------------------------------------------
             * RÉCUPÉRATION DU CHAMP output
             * ----------------------------------------------------
             */
            JsonNode output =
                    json.get("output");

            if (output == null ||
                    output.isNull()) {

                throw new RuntimeException(
                        "Le champ 'output' est absent "
                                + "de la réponse FastAPI"
                );
            }

            return output.asText();

        } catch (Exception e) {

            log.error(
                    "Erreur lors de l'appel au modèle IA "
                            + "FastAPI pour la direction {}",
                    direction,
                    e
            );

            throw new RuntimeException(
                    "Impossible de contacter le modèle "
                            + "IA FastAPI : "
                            + e.getMessage(),
                    e
            );
        }
    }

    /**
     * ============================================================
     * REQUÊTE ENVOYÉE À FASTAPI
     * ============================================================
     *
     * FastAPI attend uniquement :
     *
     * {
     *     "text": "..."
     * }
     */
    private record PredictionRequest(
            String text
    ) {
    }

    /**
     * ============================================================
     * STATISTIQUES DU DATASET
     * ============================================================
     */
    public DatasetStats getDatasetStats() {

        List<WolofPhrase> dataset =
                generator.generateDataset();

        return new DatasetStats(
                dataset.size(),
                countUniqueTokens(dataset),
                "modèle Seq2Seq avec attention"
        );
    }

    /**
     * ============================================================
     * APERÇU DU DATASET
     * ============================================================
     *
     * Méthode conservée pour compatibilité interne.
     */
    public List<WolofPhrase> getDatasetPreview(
            int limit) {

        List<WolofPhrase> dataset =
                generator.generateDataset();

        if (limit <= 0) {

            return new ArrayList<>();
        }

        if (limit >= dataset.size()) {

            return dataset;
        }

        return dataset.subList(
                0,
                limit
        );
    }

    /**
     * ============================================================
     * ÉCHANTILLON DU DATASET
     * ============================================================
     *
     * Cette méthode est appelée par :
     *
     * TransliterationController
     *
     * via :
     *
     * service.getDatasetSample(limit)
     *
     * Elle corrige donc l'erreur de compilation :
     *
     * cannot find symbol
     * method getDatasetSample(int)
     */
    public List<WolofPhrase> getDatasetSample(
            int limit) {

        return getDatasetPreview(limit);
    }

    /**
     * ============================================================
     * TOKENISATION
     * ============================================================
     *
     * Tokenisation caractère par caractère.
     *
     * Exemple :
     *
     * "xam"
     *
     * devient :
     *
     * ["x", "a", "m"]
     *
     * Cette représentation est cohérente avec le modèle
     * Seq2Seq entraîné au niveau caractère.
     */
    private List<String> tokenize(
            String text) {

        List<String> tokens =
                new ArrayList<>();

        if (text == null ||
                text.isEmpty()) {

            return tokens;
        }

        for (int i = 0;
             i < text.length();
             i++) {

            /*
             * Gestion correcte des caractères Unicode
             * représentés par des code points.
             */
            int codePoint =
                    text.codePointAt(i);

            String token =
                    new String(
                            Character.toChars(
                                    codePoint
                            )
                    );

            tokens.add(token);

            if (Character.charCount(codePoint) == 2) {
                i++;
            }
        }

        return tokens;
    }

    /**
     * ============================================================
     * CALCUL DES SCORES D'ATTENTION
     * ============================================================
     *
     * Les poids réels du mécanisme de Bahdanau sont calculés
     * dans le modèle Python.
     *
     * FastAPI ne les expose actuellement pas dans sa réponse.
     *
     * Cette méthode conserve donc le comportement attendu
     * par l'application Spring Boot.
     */
    private List<Double> computeAttentionScores(
            List<String> srcTokens,
            List<String> tgtTokens) {

        List<Double> scores =
                new ArrayList<>();

        if (srcTokens == null ||
                srcTokens.isEmpty()) {

            return scores;
        }

        /*
         * Score uniforme temporaire.
         *
         * Le modèle IA réel utilise bien le mécanisme
         * d'attention de Bahdanau côté Python.
         */
        double score =
                1.0 / srcTokens.size();

        for (int i = 0;
             i < srcTokens.size();
             i++) {

            scores.add(score);
        }

        return scores;
    }

    /**
     * ============================================================
     * COMPTAGE DES TOKENS UNIQUES
     * ============================================================
     */
    private int countUniqueTokens(
            List<WolofPhrase> dataset) {

        Set<String> uniqueTokens =
                new HashSet<>();

        for (WolofPhrase phrase : dataset) {

            if (phrase == null) {
                continue;
            }

            /*
             * Texte Latin
             */
            if (phrase.getLatin() != null) {

                for (String token :
                        tokenize(phrase.getLatin())) {

                    uniqueTokens.add(token);
                }
            }

            /*
             * Texte Ajami
             */
            if (phrase.getAjami() != null) {

                for (String token :
                        tokenize(phrase.getAjami())) {

                    uniqueTokens.add(token);
                }
            }
        }

        return uniqueTokens.size();
    }

    /**
     * ============================================================
     * DTO STATISTIQUES DATASET
     * ============================================================
     */
    public record DatasetStats(
            int totalPhrases,
            int uniqueTokens,
            String model
    ) {
    }
}