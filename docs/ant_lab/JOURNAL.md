# Journal du labo Ant

## Comment lire ce journal

Chaque expérience a deux parties. L'**hypothèse** est écrite *avant* de lancer quoi que ce soit : ce qu'on
s'attend à voir et pourquoi. Le **constat** est écrit *après* : ce qu'on a vu, les chiffres, et ce qui confirme ou
contredit l'hypothèse. L'écart entre les deux, c'est là que l'intuition se construit. Le design est dans
`docs/superpowers/specs/2026-10-30-ant-tracking-lab-design.md`.

## Notes techniques

- **Modèle Ant (sonde, 2026-10-30)** : 9 corps (`torso`, 4 `*_leg`, 4 `*_foot`), 8 articulations (hanches `*_leg`
  dans ±0,70 rad, chevilles `*_foot` dans [0,52 ; 1,75] rad en valeur absolue, posture par défaut ±0,785).
  Masse totale **0,911 kg** (torse 0,484 kg). Au repos en PD, torse à 0,435 m et bouts de pieds au sol à 0,2 mm
  près : les offsets de `nv_ant.xml` sont justes.
- **API de dynamique PhysX disponible** : matrice de masse (14×14 = 6 de base flottante + 8 articulations),
  jacobiennes (9 corps × 6 × 14), gravité et Coriolis compensés. La dynamique inverse du levier 3 est donc faisable
  avec le même modèle que le simulateur.
- Ordre de grandeur : à force 1, un couple de 7,5 N·m sur un robot de 0,9 kg, c'est beaucoup (le poids vaut ~9 N).
  À vérifier à E0 : `k_cap` crée-t-il vraiment une demande de couple au-delà de la limite du suiveur ?

## E0

### Hypothèse (écrite avant)

### Constat (écrit après)

## E1

### Hypothèse (écrite avant)

### Constat (écrit après)
