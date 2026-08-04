/**
 * Configuration de l'environnement de développement.
 * Pointe vers le serveur Spring Boot local.
 *
 * Pour la production, créer un fichier environment.prod.ts
 * avec l'URL du serveur déployé.
 */
export const environment = {
  production: false,
  apiUrl: 'http://localhost:8080/api'  // URL de l'API Spring Boot
};
