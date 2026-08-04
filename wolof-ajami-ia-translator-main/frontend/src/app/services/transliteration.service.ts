import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, tap } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import {
  TransliterationRequest,
  TransliterationResponse,
  WolofPhrase,
  DatasetStats
} from '../models/transliteration.model';

/**
 * ============================================================
 *  SERVICE ANGULAR — Communication avec Spring Boot
 * ============================================================
 *
 * Ce service centralise tous les appels HTTP vers l'API.
 * Les composants n'appellent jamais HttpClient directement,
 * ils passent toujours par ce service.
 *
 * Injection : providedIn: 'root' = disponible partout dans l'app
 * ============================================================
 */
@Injectable({
  providedIn: 'root'
})
export class TransliterationService {

  /** URL de base lue depuis environment.ts */
  private apiUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  /**
   * Translittère un texte wolof.
   * Appelle POST http://localhost:8080/api/transliterate
   *
   * @param request { text, direction }
   * @returns Observable<TransliterationResponse>
   */
  transliterate(request: TransliterationRequest): Observable<TransliterationResponse> {
    return this.http
      .post<TransliterationResponse>(`${this.apiUrl}/transliterate`, request)
      .pipe(
        tap(res => console.log(`[Translittération] ${res.processingTimeMs}ms`, res)),
        catchError(this.handleError)
      );
  }

  /**
   * Récupère un échantillon du dataset.
   * Appelle GET http://localhost:8080/api/dataset/sample?limit=N
   */
  getDatasetSample(limit = 20): Observable<WolofPhrase[]> {
    return this.http
      .get<WolofPhrase[]>(`${this.apiUrl}/dataset/sample`, { params: { limit } })
      .pipe(catchError(this.handleError));
  }

  /**
   * Récupère les statistiques du dataset.
   * Appelle GET http://localhost:8080/api/dataset/stats
   */
  getDatasetStats(): Observable<DatasetStats> {
    return this.http
      .get<DatasetStats>(`${this.apiUrl}/dataset/stats`)
      .pipe(catchError(this.handleError));
  }

  /**
   * Vérifie que le serveur Spring Boot est accessible.
   * Appelle GET http://localhost:8080/api/health
   */
  checkHealth(): Observable<string> {
    return this.http
      .get(`${this.apiUrl}/health`, { responseType: 'text' })
      .pipe(catchError(this.handleError));
  }

  /**
   * Gestion centralisée des erreurs HTTP.
   * Retourne un message lisible pour affichage dans les composants.
   */
  private handleError(error: HttpErrorResponse): Observable<never> {
    let message = 'Erreur inconnue';
    if (error.status === 0) {
      // Erreur réseau — Spring Boot probablement pas démarré
      message = 'Impossible de contacter le serveur. Vérifiez que Spring Boot tourne sur le port 8080.';
    } else if (error.status === 400) {
      // Erreur de validation (réponse JSON avec les champs invalides)
      const details = Object.values(error.error || {}).join(', ');
      message = `Requête invalide : ${details}`;
    } else if (error.status === 500) {
      message = 'Erreur interne du serveur Spring Boot.';
    }
    console.error('[TransliterationService] Erreur:', error);
    return throwError(() => new Error(message));
  }
}
