/**
 * Configuration de l'environnement de développement.
 *
 * L'application Angular utilise l'URL relative /api.
 * Le serveur Angular redirige ensuite /api vers Spring Boot
 * grâce à proxy.conf.json.
 *
 * Cela permet également d'utiliser un seul lien public
 * Cloudflare pour les tests externes.
 */
export const environment = {
  production: false,
  apiUrl: '/api'
};