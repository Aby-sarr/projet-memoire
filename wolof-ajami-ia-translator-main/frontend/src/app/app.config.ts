import { ApplicationConfig } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { routes } from './app.routes';

/**
 * Configuration globale de l'application Angular.
 *
 * - provideRouter : active le routage avec nos routes
 * - provideHttpClient : active HttpClient pour les appels à Spring Boot
 */
export const appConfig: ApplicationConfig = {
  providers: [
    provideRouter(routes),
    provideHttpClient()   // Indispensable pour que TransliterationService fonctionne
  ]
};
