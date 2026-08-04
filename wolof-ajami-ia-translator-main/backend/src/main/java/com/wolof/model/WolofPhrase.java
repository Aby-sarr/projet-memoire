package com.wolof.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Représente une paire de phrases dans le dataset wolof.
 *
 * Chaque entrée du dataset contient :
 * - La phrase en écriture latine courante (non standardisée)
 * - La traduction en orthographe officielle CLAD
 * - L'écriture Ajami (arabe wolof) si disponible
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class WolofPhrase {

    /** Phrase en écriture latine courante : ex. "Maa ngi dem ci marché bi" */
    private String latin;

    /** Phrase en orthographe officielle CLAD : ex. "Maa ngi dem ci màrse bi" */
    private String officiel;

    /** Phrase en écriture Ajami : ex. "ما نقي ديم سي مارشي بي" */
    private String ajami;
}
