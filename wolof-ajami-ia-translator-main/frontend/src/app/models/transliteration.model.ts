/**
 * ============================================================
 *  MODÈLES TYPESCRIPT
 * ============================================================
 *
 * Ces interfaces correspondent exactement aux DTOs Java Spring Boot.
 * Elles définissent la structure des données échangées via HTTP.
 * ============================================================
 */

/**
 * Corps de la requête POST /api/transliterate
 * Envoyé à Spring Boot depuis Angular
 */
export interface TransliterationRequest {
  text: string;
  /** 'lat2off' | 'off2lat' | 'lat2ajami' */
  direction: 'ajami2lat' | 'lat2ajami';
}

/**
 * Réponse reçue de Spring Boot après translittération
 */
export interface TransliterationResponse {
  input: string;
  output: string;
  direction: string;
  tokensSrc: string[];
  tokensTgt: string[];
  /** Scores entre 0 et 1 — visualisation du mécanisme d'attention */
  attentionScores: number[];
  processingTimeMs: number;
}

/**
 * Une paire de phrases du dataset wolof
 * Utilisée pour l'affichage du dataset dans DatasetViewerComponent
 */
export interface WolofPhrase {
  latin: string;
  officiel: string;
  ajami: string;
}

/**
 * Statistiques du dataset retournées par GET /api/dataset/stats
 */
export interface DatasetStats {
  totalPhrases: number;
  uniqueTokens: number;
  method: string;
}

/**
 * Paire token source + token cible avec score d'attention.
 * Utilisée uniquement en frontend pour afficher la visualisation.
 */
export interface TokenPair {
  src: string;
  tgt: string;
  score: number;
}
