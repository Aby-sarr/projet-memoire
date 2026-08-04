import { bootstrapApplication } from '@angular/platform-browser';
import { AppComponent } from './app/app.component';
import { appConfig } from './app/app.config';

/**
 * Point d'entrée de l'application Angular.
 * Lance l'application avec le composant racine et la configuration.
 */
bootstrapApplication(AppComponent, appConfig)
  .catch(err => console.error(err));
