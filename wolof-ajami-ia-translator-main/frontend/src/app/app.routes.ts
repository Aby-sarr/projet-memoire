import { Routes } from '@angular/router';
import { TransliterationComponent } from './components/transliteration/transliteration.component';
import { DatasetViewerComponent } from './components/dataset-viewer/dataset-viewer.component';

/**
 * Définition des routes de l'application.
 *
 * /          → TransliterationComponent (page principale)
 * /dataset   → DatasetViewerComponent   (exploration du dataset)
 */
export const routes: Routes = [
  { path: '',        component: TransliterationComponent, title: 'Translittération Wolof' },
  { path: 'dataset', component: DatasetViewerComponent,   title: 'Dataset Wolof' },
  { path: '**',      redirectTo: '' }  // Toute route inconnue → page principale
];
