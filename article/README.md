Contextual representation embedders [Peters et al., Pang et al.] are broadly applicable to any supervised and semi-supervised neural model, consistently benefiting NLP benchmarks: NLI [Pang et al.], dependency parsing, pos tagging, etc [Clark et al.].

Structural probes [Hewitt et al.] provide evidence that entire syntax trees are embedded implicitly in deep models' vector geometry.






Las probes evaluan la **estructura** sintáctica, no los tipos de relación

Si hay embeddings de ELMo disponibles, podemos compararlos con los míos y ver las diferencias.




Las structural probes introducidas recientemente muestran evidencias de que árboles sintácticos enteros están contenidos implícitamente en la geometría vectorial de deep models. Proponemos utilizar estas probes para validar la utilidad de downstream applications en la evaluación de herramientas sintácticas. 

Dados unos embeddings con información de estructura sintáctica implícita, si entrenamos una tarea permitiendo que modifique dichos embeddings, podríamos compararlos antes y después utilizando los probes para observar la evolución de dicha estructura sintáctica. Una degradación indicaría que su influencia en esa tarea no es muy relevante y habría sido reemplazada por otra información más útil. A su vez, esto indicaría que dicha tarea no es idónea como downstream application para evaluar herramientas sintácticas.

Para demostrar la validez/utilidad de este approach, lo aplicamos sobre dos downstream apps diferentes: 

(1) POS tagging, que se asume que no se beneficia mucho de la estructura sintáctica, por lo que esperamos observar una degradación clara de la estructura sintáctica.

(2) NLI, que se asume que se beneficia de la estructura sintáctica, por lo que esperamos observar una tendencia distinta (menos degradación, degradación desigual...).

