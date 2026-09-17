# Análisis de errores del modelo final

Modelo evaluado: `sentiment140@champion` (preprocesamiento B0 + TF-IDF unigrama+bigrama,
vocabulario de 300k términos + Logistic Regression), `test_macro_f1 = 0.8244` sobre las
240K filas de test. Del total de 42,142 errores en test, se tomó una muestra aleatoria
de 30 (semilla 42) y se clasificó cada uno según las categorías del Anexo A.6.

## Frecuencia por categoría

| Categoría | Frecuencia |
|---|---|
| other | 10 |
| elongation | 7 |
| informal | 4 |
| negation | 3 |
| contrast | 2 |
| sarcasm | 1 |
| mixed | 1 |
| intensification | 1 |
| hashtag | 1 |

## Patrones más frecuentes

1. **other (10 casos)**: la mayoría son tuits muy cortos y con poco contenido léxico de
   sentimiento explícito (`"@JaeArr *waves*"`, `"Weg stress"`, `".. Weird. The in reply
   to is one of my most used options.."`). El modelo no tiene señal n-grama fuerte a la
   que aferrarse, así que termina prediciendo la clase mayoritaria implícita en contextos
   parecidos vistos en entrenamiento. Es el error más difícil de corregir sin contexto
   adicional (el propio ser humano etiquetador probablemente usó información fuera del
   texto, como el hilo completo o el emoji renderizado).

2. **elongation (7 casos)**: alargamientos informales (`dooooog`, `sooo`, `qucik`,
   `MeaNmean.Mean`) generan tokens que no coinciden con la forma estándar de la palabra
   en el vocabulario TF-IDF (`dog`, `so`, `quick`). Como la configuración B0 no colapsa
   elongaciones, cada variante ortográfica es un término distinto y aislado, diluyendo
   la señal de sentimiento que esa palabra normalmente aportaría. Esto es consistente con
   el hallazgo de `DECISIONES.md`: la mejora de tratar elongaciones (`P_ELONGATION`,
   +0.0012 F1) es real pero pequeña porque solo un subconjunto de errores depende de esto.

3. **informal (4 casos)**: abreviaturas de chat (`u`, `im`, `thm`, `jus`, `lol`) fragmentan
   el vocabulario de la misma manera que las elongaciones — la forma abreviada rara vez
   comparte n-gramas con la forma completa, así que el clasificador pierde la asociación
   semántica aprendida sobre la palabra "canónica".

4. **negation (3 casos) + mixed (1 caso)**: el peor tipo de error para un modelo bag-of-
   n-gramas. Con bigramas el modelo capta parcialmente negaciones cortas y pegadas
   (`"not a winkk"`), pero falla cuando la negación queda separada de la palabra que
   modifica por varias palabras (`"@erin82883 NO #petewentzday..."`) o cuando el ámbito
   de la negación no es local. Un modelo secuencial (RNN/transformer) capturaría mejor
   el alcance de la negación que unigramas/bigramas independientes de posición.

5. **contrast (2 casos)**: oraciones con estructura "algo negativo... pero algo positivo"
   (o viceversa) donde ambas polaridades aparecen léxicamente y el clasificador lineal,
   al sumar contribuciones de todos los términos, termina promediando en vez de darle
   más peso a la cláusula que realmente domina el sentimiento final (típicamente la que
   viene después del conector de contraste).

## Conclusión

Los dos patrones dominantes (`other` y `elongation`, 17/30 = 57% de la muestra) apuntan
a la misma causa raíz: la representación TF-IDF con n-gramas fijos no normaliza variantes
ortográficas informales ni aporta señal cuando el texto es demasiado corto o carece de
léxico de sentimiento explícito. Los errores de `negation`/`contrast`/`mixed` (6/30 = 20%)
son estructurales: dependen del orden y del alcance sintáctico, algo que un modelo lineal
sobre bag-of-n-gramas no puede modelar. Ambas limitaciones son esperables dado el diseño
elegido (B0 + TF-IDF + Logistic Regression) y están documentadas como líneas de mejora
futura (normalización de elongaciones, modelos secuenciales) más que como fallos del
pipeline en sí.
