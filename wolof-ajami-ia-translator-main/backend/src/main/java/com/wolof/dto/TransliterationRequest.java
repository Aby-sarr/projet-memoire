package com.wolof.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import lombok.Data;

/**
 * Objet reçu depuis Angular dans le corps de la requête POST.
 *
 * Exemple JSON envoyé par Angular :
 *
 * {
 *   "text": "jàmm",
 *   "direction": "lat2ajami"
 * }
 *
 * Les directions disponibles sont :
 *
 * - lat2off   : Latin courant → Orthographe officielle CLAD
 * - off2lat   : Orthographe officielle CLAD → Latin courant
 * - lat2ajami : Latin → Écriture Ajami
 * - ajami2lat : Écriture Ajami → Latin
 */
@Data
public class TransliterationRequest {

    /**
     * Texte wolof à translittérer.
     */
    @NotBlank(message = "Le texte ne peut pas être vide")
    private String text;

    /**
     * Direction de translittération.
     */
    @Pattern(
        regexp = "lat2off|off2lat|lat2ajami|ajami2lat",
        message = "Direction invalide. Valeurs: lat2off, off2lat, lat2ajami, ajami2lat"
    )
    private String direction;
}