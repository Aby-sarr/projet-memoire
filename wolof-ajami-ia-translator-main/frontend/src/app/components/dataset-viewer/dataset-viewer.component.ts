import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { TransliterationService } from '../../services/transliteration.service';
import { WolofPhrase, DatasetStats } from '../../models/transliteration.model';

/**
 * Composant d'exploration du dataset.
 * Permet de visualiser les phrases générées et les stats.
 * Route : /dataset
 */
@Component({
  selector: 'app-dataset-viewer',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="page">
      <h1>Dataset Wolof</h1>
      <p class="subtitle">Aperçu du dataset généré (5000 phrases)</p>

      <!-- Stats -->
      <div class="stats-grid" *ngIf="stats">
        <div class="stat-card">
          <div class="stat-value">{{ stats.totalPhrases | number }}</div>
          <div class="stat-label">Phrases totales</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ stats.uniqueTokens | number }}</div>
          <div class="stat-label">Tokens uniques</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">3</div>
          <div class="stat-label">Scripts (Latin, CLAD, Ajami)</div>
        </div>
      </div>

      <!-- Contrôle du nombre d'exemples -->
      <div class="controls">
        <label>Afficher : </label>
        <select [(ngModel)]="limit" (change)="loadSample()">
          <option [value]="10">10 phrases</option>
          <option [value]="25">25 phrases</option>
          <option [value]="50">50 phrases</option>
          <option [value]="100">100 phrases</option>
        </select>
        <button class="btn" (click)="loadSample()">Actualiser</button>
      </div>

      <!-- Tableau des phrases -->
      <div class="table-wrap">
        <table *ngIf="!loading && phrases.length > 0">
          <thead>
            <tr>
              <th>#</th>
              <th>Latin courant</th>
              <th>Officiel CLAD</th>
              <th>Ajami</th>
            </tr>
          </thead>
          <tbody>
            <tr *ngFor="let phrase of phrases; let i = index">
              <td class="num">{{ i + 1 }}</td>
              <td>{{ phrase.latin }}</td>
              <td>{{ phrase.officiel }}</td>
              <td class="ajami">{{ phrase.ajami }}</td>
            </tr>
          </tbody>
        </table>
        <div *ngIf="loading" class="loading">Chargement du dataset...</div>
      </div>
    </div>
  `,
  styles: [`
    .page { max-width: 1000px; margin: 0 auto; padding: 2rem 1.5rem; font-family: 'Segoe UI', sans-serif; }
    h1 { font-size: 1.5rem; margin-bottom: 4px; }
    .subtitle { color: #6c757d; margin-bottom: 1.5rem; }
    .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 1.5rem; }
    .stat-card { background: #f8f9fb; border: 1px solid #e2e6ea; border-radius: 10px; padding: 1rem; text-align: center; }
    .stat-value { font-size: 1.8rem; font-weight: 600; color: #185FA5; }
    .stat-label { font-size: 0.8rem; color: #6c757d; margin-top: 2px; }
    .controls { display: flex; align-items: center; gap: 10px; margin-bottom: 1rem; }
    select { padding: 6px 10px; border: 1px solid #e2e6ea; border-radius: 6px; }
    .btn { padding: 6px 14px; background: #185FA5; color: white; border: none; border-radius: 6px; cursor: pointer; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    th { background: #f8f9fb; padding: 10px 12px; text-align: left; border-bottom: 2px solid #e2e6ea; font-weight: 600; color: #495057; }
    td { padding: 8px 12px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }
    tr:hover td { background: #f8f9fb; }
    .num { color: #adb5bd; width: 40px; }
    .ajami { direction: rtl; font-size: 1rem; }
    .loading { text-align: center; padding: 2rem; color: #6c757d; }
  `]
})
export class DatasetViewerComponent implements OnInit {
  phrases: WolofPhrase[] = [];
  stats: DatasetStats | null = null;
  loading = false;
  limit = 25;

  constructor(private service: TransliterationService) {}

  ngOnInit(): void {
    this.loadSample();
    this.service.getDatasetStats().subscribe(s => this.stats = s);
  }

  loadSample(): void {
    this.loading = true;
    this.service.getDatasetSample(this.limit).subscribe({
      next: data => { this.phrases = data; this.loading = false; },
      error: ()  => { this.loading = false; }
    });
  }
}
