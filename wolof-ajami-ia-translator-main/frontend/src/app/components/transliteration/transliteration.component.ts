import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { TransliterationService } from '../../services/transliteration.service';
import {
  TransliterationResponse,
  TokenPair
} from '../../models/transliteration.model';

/**
 * ============================================================
 *  COMPOSANT PRINCIPAL — Interface de translittération
 * ============================================================
 *
 * Gère :
 *  - La saisie du texte
 *  - Le choix de direction
 *  - L'affichage du résultat
 *  - La visualisation du mécanisme d'attention
 *  - Les exemples rapides cliquables
 *
 * Directions disponibles :
 *  - Ajami → Latin
 *  - Latin → Ajami
 * ============================================================
 */
@Component({
  selector: 'app-transliteration',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './transliteration.component.html',
  styleUrls: ['./transliteration.component.scss']
})
export class TransliterationComponent implements OnInit {

  // -------------------------------------------------------
  // STATE — Variables liées au template HTML via [(ngModel)]
  // -------------------------------------------------------

  /** Texte saisi par l'utilisateur */
  inputText = '';

  /** Résultat retourné par Spring Boot */
  outputText = '';

  /** Direction sélectionnée */
  direction: 'ajami2lat' | 'lat2ajami' = 'ajami2lat';

  /** Paires token source/cible avec leurs scores d'attention */
  tokenPairs: TokenPair[] = [];

  /** Temps de traitement affiché */
  processingTimeMs = 0;

  /** true pendant l'appel HTTP */
  loading = false;

  /** Message d'erreur à afficher */
  errorMessage: string | null = null;

  /** true si Spring Boot est accessible */
  serverOnline = false;

  // -------------------------------------------------------
  // EXEMPLES RAPIDES par direction
  // -------------------------------------------------------

  readonly examples: Record<
    string,
    { label: string; text: string }[]
  > = {

    // -----------------------------------------------------
    // AJAMI → LATIN
    // -----------------------------------------------------
    ajami2lat: [
      { label: 'Jamm',     text: 'جامم' },
      { label: 'Jërëjëf',  text: 'جەرەجەف' },
      { label: 'Kër',      text: 'كەر' },
      { label: 'Nit',      text: 'نيت' },
      { label: 'Jàng',     text: 'جاڠ' }
    ],

    // -----------------------------------------------------
    // LATIN → AJAMI
    // -----------------------------------------------------
    lat2ajami: [
      { label: 'Jàmm',     text: 'jàmm' },
      { label: 'Jërëjëf',  text: 'jërejëf' },
      { label: 'Kër',      text: 'kër' },
      { label: 'Nit',      text: 'nit' },
      { label: 'Jàng',     text: 'jàng' }
    ]
  };

  constructor(
    private transliterationService: TransliterationService
  ) {}

  // -------------------------------------------------------
  // INITIALISATION
  // -------------------------------------------------------

  ngOnInit(): void {

    // Vérifier la connexion au démarrage
    this.transliterationService.checkHealth().subscribe({
      next: () => {
        this.serverOnline = true;
      },
      error: () => {
        this.serverOnline = false;
      }
    });
  }

  // -------------------------------------------------------
  // EXEMPLES
  // -------------------------------------------------------

  /** Retourne les exemples de la direction courante */
  get currentExamples() {
    return this.examples[this.direction] || [];
  }

  // -------------------------------------------------------
  // LABEL DE LA DIRECTION
  // -------------------------------------------------------

  /** Label lisible de la direction courante */
  get directionLabel(): string {

    const labels: Record<string, string> = {
      ajami2lat: 'Ajami → Latin',
      lat2ajami: 'Latin → Ajami (arabe wolof)'
    };

    return labels[this.direction];
  }

  // -------------------------------------------------------
  // TRANSLITTÉRATION
  // -------------------------------------------------------

  /**
   * Appelé quand l'utilisateur clique sur "Translittérer".
   *
   * Envoie la requête à Spring Boot via le service.
   */
  onTransliterate(): void {

    // Ne rien faire si le champ est vide
    // ou si une requête est déjà en cours
    if (!this.inputText.trim() || this.loading) {
      return;
    }

    this.loading = true;
    this.errorMessage = null;
    this.outputText = '';
    this.tokenPairs = [];

    this.transliterationService.transliterate({
      text: this.inputText.trim(),
      direction: this.direction
    }).subscribe({

      // ---------------------------------------------------
      // SUCCÈS
      // ---------------------------------------------------
      next: (res: TransliterationResponse) => {

        this.outputText = res.output;

        this.processingTimeMs = res.processingTimeMs;

        // Construire les paires token source/cible
        // pour la visualisation du mécanisme d'attention
        this.tokenPairs = res.tokensSrc.map((src, i) => ({
          src,
          tgt: res.tokensTgt[i] ?? '',
          score: res.attentionScores[i] ?? 0.5
        }));

        this.loading = false;
      },

      // ---------------------------------------------------
      // ERREUR
      // ---------------------------------------------------
      error: (err: Error) => {

        this.errorMessage = err.message;

        this.loading = false;
      }
    });
  }

  // -------------------------------------------------------
  // CHARGEMENT D'UN EXEMPLE
  // -------------------------------------------------------

  /** Charge un exemple dans le champ de saisie */
  useExample(text: string): void {

    this.inputText = text;

    this.outputText = '';

    this.tokenPairs = [];

    this.errorMessage = null;

    this.processingTimeMs = 0;
  }

  // -------------------------------------------------------
  // RÉINITIALISATION
  // -------------------------------------------------------

  /** Réinitialise le formulaire */
  onReset(): void {

    this.inputText = '';

    this.outputText = '';

    this.tokenPairs = [];

    this.errorMessage = null;

    this.processingTimeMs = 0;
  }

  // -------------------------------------------------------
  // VISUALISATION DE L'ATTENTION
  // -------------------------------------------------------

  /** Convertit un score (0–1) en largeur CSS */
  barWidth(score: number): string {

    return `${Math.round(10 + score * 70)}px`;
  }

  /** Convertit un score en opacité CSS */
  barOpacity(score: number): number {

    return 0.25 + score * 0.75;
  }

  /** Retourne la couleur de la barre selon le score */
  barColor(score: number): string {

    if (score >= 0.8) {
      return '#185FA5';
    }

    if (score >= 0.5) {
      return '#4A90D9';
    }

    return '#A0C4E8';
  }
}
```
