package com.wolof.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import lombok.Data;

/**
 * Objet reçu depuis Angular dans le corps de la requête POST.
 *
 * Contient le texte à translittérer ainsi que la direction
 * de translittération sélectionnée dans l'interface web.
 *
 * Directions disponibles :
 * - lat2ajami : Latin → Ajami
 * - ajami2lat : Ajami → Latin
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
     *
     * lat2ajami : Latin → Ajami
     * ajami2lat : Ajami → Latin
     */
    @Pattern(
        regexp = "lat2ajami|ajami2lat",
        message = "Direction invalide. Valeurs: lat2ajami, ajami2lat"
    )
    private String direction;
}