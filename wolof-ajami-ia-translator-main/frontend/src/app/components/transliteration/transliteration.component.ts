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
 *  - La saisie du texte wolof
 *  - Le choix de direction
 *  - L'affichage du résultat
 *  - La visualisation du mécanisme d'attention
 *  - Les exemples rapides cliquables
 * ============================================================
 */
@Component({
  selector: 'app-transliteration',
  standalone: true,
  imports: [CommonModule, FormsModule],  // FormsModule pour [(ngModel)]
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
  direction: 'lat2off' | 'off2lat' | 'lat2ajami' = 'lat2off';

  /** Paires token source/cible avec leurs scores d'attention */
  tokenPairs: TokenPair[] = [];

  /** Temps de traitement affiché */
  processingTimeMs = 0;

  /** true pendant l'appel HTTP */
  loading = false;

  /** Message d'erreur à afficher (null = pas d'erreur) */
  errorMessage: string | null = null;

  /** true si Spring Boot est accessible */
  serverOnline = false;

  // -------------------------------------------------------
  // EXEMPLES RAPIDES par direction
  // -------------------------------------------------------
  readonly examples: Record<string, { label: string; text: string }[]> = {
    lat2off: [
      { label: 'Je vais au marché', text: 'Maa ngi dem ci marché bi' },
      { label: 'Merci beaucoup',    text: 'Jerejef bu baax' },
      { label: 'L\'enfant apprend', text: 'Xale bi jàng ci école bi' },
      { label: 'Il travaille',      text: 'Mu ngi liggéey ci bureau bi' },
      { label: 'Bonne nuit',        text: 'Nelaw naa bu neex' },
    ],
    off2lat: [
      { label: 'Je vais au marché', text: 'Maa ngi dem ci màrse bi' },
      { label: 'Il travaille',      text: 'Mu ngi liggéey ci biiro bi' },
      { label: 'L\'enfant apprend', text: 'Xale bi jàng ci ekol bi' },
    ],
    lat2ajami: [
      { label: 'Je vais au marché', text: 'Maa ngi dem ci marché bi' },
      { label: 'Merci',             text: 'Jerejef' },
      { label: 'L\'eau',            text: 'ndox mi' },
    ]
  };

  constructor(private transliterationService: TransliterationService) {}

  ngOnInit(): void {
    // Vérifier la connexion au démarrage
    this.transliterationService.checkHealth().subscribe({
      next: () => { this.serverOnline = true; },
      error: () => { this.serverOnline = false; }
    });
  }

  /** Retourne les exemples de la direction courante */
  get currentExamples() {
    return this.examples[this.direction] || [];
  }

  /** Label lisible de la direction courante */
  get directionLabel(): string {
    const labels: Record<string, string> = {
      lat2off:   'Latin courant → Officiel CLAD',
      off2lat:   'Officiel CLAD → Latin courant',
      lat2ajami: 'Latin → Ajami (arabe wolof)'
    };
    return labels[this.direction];
  }

  /**
   * Appelé quand l'utilisateur clique sur "Translittérer".
   * Envoie la requête à Spring Boot via le service.
   */
  onTransliterate(): void {
    if (!this.inputText.trim() || this.loading) return;

    this.loading = true;
    this.errorMessage = null;
    this.outputText = '';
    this.tokenPairs = [];

    this.transliterationService.transliterate({
      text: this.inputText.trim(),
      direction: this.direction
    }).subscribe({
      next: (res: TransliterationResponse) => {
        this.outputText      = res.output;
        this.processingTimeMs = res.processingTimeMs;
        // Construire les paires token pour la visualisation d'attention
        this.tokenPairs = res.tokensSrc.map((src, i) => ({
          src,
          tgt:   res.tokensTgt[i]   ?? '',
          score: res.attentionScores[i] ?? 0.5
        }));
        this.loading = false;
      },
      error: (err: Error) => {
        this.errorMessage = err.message;
        this.loading = false;
      }
    });
  }

  /** Charge un exemple dans le champ de saisie */
  useExample(text: string): void {
    this.inputText = text;
    this.outputText = '';
    this.tokenPairs = [];
    this.errorMessage = null;
  }

  /** Réinitialise le formulaire */
  onReset(): void {
    this.inputText = '';
    this.outputText = '';
    this.tokenPairs = [];
    this.errorMessage = null;
    this.processingTimeMs = 0;
  }

  /** Convertit un score (0–1) en largeur CSS pour la barre d'attention */
  barWidth(score: number): string {
    return `${Math.round(10 + score * 70)}px`;
  }

  /** Convertit un score en opacité CSS */
  barOpacity(score: number): number {
    return 0.25 + score * 0.75;
  }

  /** Retourne la couleur de la barre selon le score */
  barColor(score: number): string {
    if (score >= 0.8) return '#185FA5';  // Bleu fort = haute attention
    if (score >= 0.5) return '#4A90D9';  // Bleu moyen
    return '#A0C4E8';                    // Bleu clair = faible attention
  }
}
