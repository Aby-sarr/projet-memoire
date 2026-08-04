# Wolof Translittération — Guide de démarrage rapide

## Prérequis
- Java 17+ installé → `java -version`
- Node.js 18+ installé → `node -v`
- Angular CLI installé → `npm install -g @angular/cli`

---

## 1. Démarrer Spring Boot

```bash
cd backend

# Linux/Mac
./gradlew bootRun

# Windows
gradlew.bat bootRun
```

Le serveur démarre sur **http://localhost:8080**

### Tester que ça marche :
```bash
curl http://localhost:8080/api/health
# → OK — Serveur Wolof opérationnel

curl -X POST http://localhost:8080/api/transliterate \
  -H "Content-Type: application/json" \
  -d '{"text":"Maa ngi dem ci marché bi","direction":"lat2off"}'
```

---

## 2. Démarrer Angular

```bash
cd frontend

# Installer les dépendances (une seule fois)
npm install

# Démarrer le serveur de développement
ng serve
```

L'application est accessible sur **http://localhost:4200**

---

## Structure du projet

```
wolof-transliteration/
├── backend/                          ← Spring Boot (Gradle)
│   ├── build.gradle                  ← Dépendances Java
│   ├── settings.gradle
│   └── src/main/java/com/wolof/
│       ├── WolofApplication.java     ← Point d'entrée Spring Boot
│       ├── config/CorsConfig.java    ← Autorise Angular (port 4200)
│       ├── controller/
│       │   ├── TransliterationController.java  ← Endpoints REST
│       │   └── GlobalExceptionHandler.java     ← Gestion erreurs
│       ├── service/
│       │   └── TransliterationService.java     ← Logique métier
│       ├── dataset/
│       │   └── WolofDatasetGenerator.java      ← 5000 phrases wolof
│       ├── dto/
│       │   ├── TransliterationRequest.java
│       │   └── TransliterationResponse.java
│       └── model/
│           └── WolofPhrase.java
│
└── frontend/                         ← Angular 17
    ├── package.json
    ├── angular.json
    └── src/
        ├── main.ts                   ← Point d'entrée Angular
        ├── index.html
        ├── styles.scss               ← Styles globaux
        ├── environments/
        │   └── environment.ts        ← URL de l'API Spring Boot
        └── app/
            ├── app.component.ts      ← Composant racine + navigation
            ├── app.config.ts         ← Providers (HttpClient, Router)
            ├── app.routes.ts         ← Routes / et /dataset
            ├── models/
            │   └── transliteration.model.ts    ← Interfaces TypeScript
            ├── services/
            │   └── transliteration.service.ts  ← Appels HTTP vers Spring Boot
            └── components/
                ├── transliteration/             ← Page principale
                │   ├── transliteration.component.ts
                │   ├── transliteration.component.html
                │   └── transliteration.component.scss
                └── dataset-viewer/             ← Page dataset
                    └── dataset-viewer.component.ts
```

---

## Endpoints API Spring Boot

| Méthode | URL | Description |
|---------|-----|-------------|
| GET  | `/api/health`            | Vérifier que le serveur tourne |
| POST | `/api/transliterate`     | Translittérer un texte |
| GET  | `/api/dataset/sample`    | Voir N phrases du dataset |
| GET  | `/api/dataset/stats`     | Statistiques du dataset |

### Corps POST /api/transliterate
```json
{
  "text": "Maa ngi dem ci marché bi",
  "direction": "lat2off"
}
```
Directions disponibles : `lat2off` | `off2lat` | `lat2ajami`

---

## Prochaine étape — Intégrer le vrai modèle Seq2Seq

Quand ton modèle sera entraîné, remplace dans `TransliterationService.java` :
```java
// AVANT (règles)
String outputText = generator.toOfficiel(request.getText());

// APRÈS (modèle DJL)
String outputText = seq2seqModel.translate(request.getText(), request.getDirection());
```
