import { Component } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';

/**
 * Composant racine de l'application.
 * Contient la navigation et le <router-outlet> qui affiche
 * le composant correspondant à la route active.
 */
@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  template: `
    <!-- Barre de navigation -->
    <nav class="navbar">
      <span class="brand">🗣 Wolof Translittération</span>
      <div class="nav-links">
        <a routerLink="/"        routerLinkActive="active" [routerLinkActiveOptions]="{exact: true}">Translittérer</a>
        <a routerLink="/dataset" routerLinkActive="active">Dataset</a>
      </div>
    </nav>

    <!-- Zone de contenu : le composant actif s'affiche ici -->
    <main>
      <router-outlet />
    </main>
  `,
  styles: [`
    .navbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 2rem;
      background: #185FA5;
      color: white;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .brand { font-size: 1rem; font-weight: 600; }
    .nav-links { display: flex; gap: 1.5rem; }
    .nav-links a {
      color: rgba(255,255,255,0.85);
      text-decoration: none;
      font-size: 0.9rem;
      padding: 4px 0;
      border-bottom: 2px solid transparent;
      transition: all 0.15s;
    }
    .nav-links a:hover, .nav-links a.active {
      color: white;
      border-bottom-color: white;
    }
    main { min-height: calc(100vh - 52px); }
  `]
})
export class AppComponent {}
