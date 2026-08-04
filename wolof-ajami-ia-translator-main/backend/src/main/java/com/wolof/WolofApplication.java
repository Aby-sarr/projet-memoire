package com.wolof;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Point d'entrée principal de l'application Spring Boot.
 * Lance le serveur sur http://localhost:8080
 *
 * Pour démarrer : ./gradlew bootRun
 */
@SpringBootApplication
public class WolofApplication {

    public static void main(String[] args) {
        SpringApplication.run(WolofApplication.class, args);
    }
}
