package com.wolof.controller;

import com.wolof.dto.TransliterationRequest;
import com.wolof.dto.TransliterationResponse;
import com.wolof.model.WolofPhrase;
import com.wolof.service.TransliterationService;
import jakarta.validation.Valid;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * ============================================================
 *  CONTRÔLEUR REST — Point d'entrée HTTP
 * ============================================================
 *
 * Expose les endpoints que Angular va appeler.
 *
 * Base URL : http://localhost:8080/api
 *
 * Endpoints disponibles :
 *   POST /api/transliterate        → Translittérer un texte
 *   GET  /api/dataset/sample       → Voir des exemples du dataset
 *   GET  /api/dataset/stats        → Statistiques du dataset
 *   GET  /api/health               → Vérifier que le serveur tourne
 * ============================================================
 */
@RestController
@RequestMapping("/api")
@Slf4j
public class TransliterationController {

    private final TransliterationService service;

    public TransliterationController(TransliterationService service) {
        this.service = service;
    }

    /**
     * POST /api/transliterate
     *
     * Reçoit un texte wolof et retourne sa translittération.
     *
     * Corps de la requête (JSON) :
     * {
     *   "text": "Maa ngi dem ci marché bi",
     *   "direction": "lat2off"
     * }
     *
     * @Valid déclenche la validation des annotations dans TransliterationRequest
     */
    @PostMapping("/transliterate")
    public ResponseEntity<TransliterationResponse> transliterate(
            @Valid @RequestBody TransliterationRequest request) {

        log.info("→ POST /transliterate | text='{}' | dir='{}'",
                 request.getText(), request.getDirection());

        TransliterationResponse response = service.transliterate(request);
        return ResponseEntity.ok(response);
    }

    /**
     * GET /api/dataset/sample?limit=20
     *
     * Retourne un aperçu du dataset généré.
     * Utile pour vérifier les données depuis Angular.
     *
     * @param limit nombre de phrases à retourner (défaut: 20)
     */
    @GetMapping("/dataset/sample")
    public ResponseEntity<List<WolofPhrase>> getDatasetSample(
            @RequestParam(defaultValue = "20") int limit) {

        log.info("→ GET /dataset/sample?limit={}", limit);
        return ResponseEntity.ok(service.getDatasetSample(limit));
    }

    /**
     * GET /api/dataset/stats
     *
     * Retourne les statistiques du dataset (nombre de phrases, tokens, etc.)
     */
    @GetMapping("/dataset/stats")
    public ResponseEntity<TransliterationService.DatasetStats> getStats() {
        log.info("→ GET /dataset/stats");
        return ResponseEntity.ok(service.getDatasetStats());
    }

    /**
     * GET /api/health
     *
     * Endpoint simple pour vérifier que le serveur est démarré.
     * Angular peut appeler ceci au démarrage pour confirmer la connexion.
     */
    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("OK — Serveur Wolof opérationnel");
    }
}
