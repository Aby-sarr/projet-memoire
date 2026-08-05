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
import java.util.List;

/**
 * ============================================================
 * SERVICE DE TRANSLITTÉRATION
 * ============================================================
 *
 * Ce service assure la communication entre Angular,
 * Spring Boot et le modèle IA Python/FastAPI.
 *
 * Architecture :
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
     * Adresse de l'API FastAPI.
     */
    private static final String AI_API_URL =
            "http://127.0.0.1:5000";

    public TransliterationService(
            WolofDatasetGenerator generator) {

        this.generator = generator;

        this.restClient = RestClient.builder()
                .baseUrl(AI_API_URL)
                .build();

        this.objectMapper = new ObjectMapper();
    }

    /**
     * Point d'entrée principal de la translittération.
     */
    public TransliterationResponse transliterate(
            TransliterationRequest request) {

        long startTime = System.currentTimeMillis();

        String text = request.getText();
        String direction = request.getDirection();

        log.debug(
                "Translittération : '{}' [{}]",
                text,
                direction
        );

        /*
         * 1. Tokenisation du texte source.
         */
        List<String> srcTokens = tokenize(text);

        /*
         * 2. Translittération selon la direction demandée.
         */
        String outputText;

        switch (direction) {

            case "lat2ajami":

                /*
                 * Latin → Ajami
                 * Utilise le modèle IA.
                 */
                outputText = callAiModel(
                        text,
                        "lat2ajami"
                );

                break;

            case "ajami2lat":

                /*
                 * Ajami → Latin
                 * Utilise le modèle IA inverse.
                 */
                outputText = callAiModel(
                        text,
                        "ajami2lat"
                );

                break;

            case "lat2off":

                /*
                 * Latin courant → orthographe officielle CLAD.
                 */
                outputText =
                        generator.toOfficiel(text);

                break;

            case "off2lat":

                /*
                 * Orthographe officielle CLAD → Latin courant.
                 */
                outputText =
                        generator.toLatinFromOfficiel(text);

                break;

            default:

                throw new IllegalArgumentException(
                        "Direction invalide : " + direction
                );
        }

        /*
         * 3. Tokenisation du résultat.
         */
        List<String> tgtTokens = tokenize(outputText);

        /*
         * 4. Calcul des scores d'attention.
         *
         * Les vrais poids Bahdanau ne sont pas encore
         * retournés par FastAPI.
         */
        List<Double> attentionScores =
                computeAttentionScores(
                        srcTokens,
                        tgtTokens
                );

        long processingTime =
                System.currentTimeMillis() - startTime;

        log.debug(
                "Résultat : '{}' en {} ms",
                outputText,
                processingTime
        );

        /*
         * 5. Réponse envoyée à Angular.
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
     * Appelle :
     *
     * POST http://127.0.0.1:5000/predict
     *
     * avec :
     *
     * {
     *     "text": "jamm",
     *     "direction": "lat2ajami"
     * }
     *
     * ou :
     *
     * {
     *     "text": "جامم",
     *     "direction": "ajami2lat"
     * }
     *
     * La réponse attendue contient notamment :
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

            String response =
                    restClient.post()
                            .uri("/predict")
                            .contentType(
                                    MediaType.APPLICATION_JSON
                            )
                            .body(
                                    new PredictionRequest(
                                            text,
                                            direction
                                    )
                            )
                            .retrieve()
                            .body(String.class);

            if (response == null ||
                    response.isBlank()) {

                throw new RuntimeException(
                        "Réponse vide de FastAPI"
                );
            }

            JsonNode json =
                    objectMapper.readTree(response);

            JsonNode output =
                    json.get("output");

            if (output == null ||
                    output.isNull()) {

                throw new RuntimeException(
                        "Le champ 'output' est absent de la réponse FastAPI"
                );
            }

            return output.asText();

        } catch (Exception e) {

            log.error(
                    "Erreur lors de l'appel au modèle IA FastAPI pour la direction {}",
                    direction,
                    e
            );

            throw new RuntimeException(
                    "Impossible de contacter le modèle IA FastAPI : "
                            + e.getMessage(),
                    e
            );
        }
    }

    /**
     * Requête envoyée à FastAPI.
     */
    private record PredictionRequest(
            String text,
            String direction
    ) {}

    /**
     * Retourne les statistiques du dataset.
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
     * Retourne un aperçu du dataset.
     */
    public List<WolofPhrase> getDatasetSample(
            int limit) {

        List<WolofPhrase> dataset =
                generator.generateDataset();

        return dataset.subList(
                0,
                Math.min(limit, dataset.size())
        );
    }

    /**
     * Tokenise le texte selon les espaces.
     */
    private List<String> tokenize(String text) {

        if (text == null ||
                text.trim().isEmpty()) {

            return List.of();
        }

        return List.of(
                text.trim().split("\\s+")
        );
    }

    /**
     * Calcule temporairement les scores d'attention.
     *
     * Les vrais poids Bahdanau pourront être récupérés
     * directement depuis FastAPI dans une prochaine étape.
     */
    private List<Double> computeAttentionScores(
            List<String> src,
            List<String> tgt) {

        List<Double> scores =
                new ArrayList<>();

        int maxLen =
                Math.max(
                        src.size(),
                        tgt.size()
                );

        for (int i = 0; i < maxLen; i++) {

            String srcTok =
                    i < src.size()
                            ? src.get(i)
                            : "";

            String tgtTok =
                    i < tgt.size()
                            ? tgt.get(i)
                            : "";

            scores.add(
                    generator.computeAttentionScore(
                            srcTok,
                            tgtTok
                    )
            );
        }

        return scores;
    }

    /**
     * Compte les tokens uniques du dataset.
     */
    private int countUniqueTokens(
            List<WolofPhrase> dataset) {

        return (int) dataset.stream()
                .flatMap(
                        p -> List.of(
                                p.getLatin()
                                        .split("\\s+")
                        ).stream()
                )
                .distinct()
                .count();
    }

    /**
     * Structure contenant les statistiques du dataset.
     */
    public record DatasetStats(
            int totalPhrases,
            int uniqueTokens,
            String method
    ) {}
}