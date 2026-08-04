package com.wolof.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import lombok.Data;

/**
 * Objet reçu depuis Angular dans le corps de la requête POST.
 *
 * Exemple JSON envoyé par Angular :
 * {
 *   "text": "Maa ngi dem ci marché bi",
 *   "direction": "lat2off"
 * }
 */
@Data  // Lombok génère automatiquement getters, setters, toString, equals
public class TransliterationRequest {

    /**
     * Le texte wolof à translittérer.
     * @NotBlank : refuse les textes vides ou null
     */
    @NotBlank(message = "Le texte ne peut pas être vide")
    private String text;

    /**
     * Direction de translittération :
     * - "lat2off" : Latin courant → Orthographe officielle CLAD
     * - "off2lat" : Orthographe officielle CLAD → Latin courant
     * - "lat2ajami" : Latin → Écriture Ajami (arabe wolof)
     */
    @Pattern(regexp = "lat2off|off2lat|lat2ajami",
             message = "Direction invalide. Valeurs: lat2off, off2lat, lat2ajami")
    private String direction;
}
