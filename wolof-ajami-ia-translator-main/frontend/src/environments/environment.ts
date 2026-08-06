/**
 * Configuration de l'environnement de développement.
 *
 * L'application Angular peut être utilisée :
 * - localement avec Spring Boot sur localhost:8080
 * - via Cloudflare Tunnel pour les testeurs externes
 */

export const environment = {
  production: false,

  /**
   * URL publique de Spring Boot.
   *
   * Cloudflare Tunnel redirige cette URL vers :
   * http://localhost:8080
   */
  apiUrl: 'https://types-yeast-mini-transition.trycloudflare.com/api'
};