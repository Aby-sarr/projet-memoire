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
 * Le backend Spring Boot sert d'intermédiaire entre Angular
 * et le modèle IA Python/FastAPI.
 *
 * Architecture :
 *
 * Angular
 *    ↓
 * Spring Boot
 *    ↓
 * FastAPI : http://127.0.0.1:5000/predict
 *    ↓
 * PyTorch / best_model.pt
 *
 * Les directions lat2off et off2lat restent gérées par
 * WolofDatasetGenerator.
 *
 * La direction lat2ajami utilise désormais le vrai modèle IA.
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
     * Point d'entrée principal.
     */
    public TransliterationResponse transliterate(
            TransliterationRequest request) {

        long startTime = System.currentTimeMillis();

        String text = request.getText();
        String direction = request.getDirection();

        log.debug(
                "Translittération: '{}' [{}]",
                text,
                direction
        );

        // ----------------------------------------------------
        // 1. Texte source
        // ----------------------------------------------------

        List<String> srcTokens = tokenize(text);

        // ----------------------------------------------------
        // 2. Translittération selon la direction
        // ----------------------------------------------------

        String outputText;

        switch (direction) {

            case "lat2ajami":

                outputText = callAiModel(text);

                break;

            case "lat2off":

                outputText =
                        generator.toOfficiel(text);

                break;

            case "off2lat":

                outputText =
                        generator.toLatinFromOfficiel(text);

                break;

            default:

                outputText = text;
        }

        // ----------------------------------------------------
        // 3. Tokens cible
        // ----------------------------------------------------

        List<String> tgtTokens =
                tokenize(outputText);

        // ----------------------------------------------------
        // 4. Attention
        // ----------------------------------------------------
        //
        // Pour le moment, les vrais poids d'attention
        // ne sont pas encore renvoyés par FastAPI.
        //
        // On conserve donc le mécanisme actuel afin de
        // ne pas casser Angular.
        // ----------------------------------------------------

        List<Double> attentionScores =
                computeAttentionScores(
                        srcTokens,
                        tgtTokens
                );

        long processingTime =
                System.currentTimeMillis() - startTime;

        log.debug(
                "Résultat IA: '{}' en {}ms",
                outputText,
                processingTime
        );

        // ----------------------------------------------------
        // 5. Réponse vers Angular
        // ----------------------------------------------------

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
     * ========================================================
     * APPEL DU MODÈLE IA PYTHON
     * ========================================================
     *
     * Appelle :
     *
     * POST http://127.0.0.1:5000/predict
     *
     * avec :
     *
     * {
     *     "text": "ndank"
     * }
     *
     * et récupère :
     *
     * {
     *     "input": "ndank",
     *     "output": "ندانك",
     *     "direction": "lat2ajami"
     * }
     */
    private String callAiModel(String text) {

        try {

            String response =
                    restClient.post()
                            .uri("/predict")
                            .contentType(
                                    MediaType.APPLICATION_JSON
                            )
                            .body(
                                    new PredictionRequest(text)
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
                    "Erreur lors de l'appel au modèle IA FastAPI",
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
            String text
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
                "modèle IA Seq2Seq pour lat2ajami"
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

    // --------------------------------------------------------
    // MÉTHODES PRIVÉES
    // --------------------------------------------------------

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
     * Conservation temporaire de l'attention simulée.
     *
     * IMPORTANT :
     * Cette méthode sera remplacée ensuite par les vrais
     * poids Bahdanau renvoyés par FastAPI.
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
     * Compte les tokens uniques.
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
     * Statistiques du dataset.
     */
    public record DatasetStats(
            int totalPhrases,
            int uniqueTokens,
            String method
    ) {}
}