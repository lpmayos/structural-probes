.conllu copied from example/data/ with sentences containing tokens with dots in id have been removed. i.e.
 
    ...
    7	and	and	CCONJ	CC	_	8	cc	8:cc|8.1:cc	_
    8	500	500	NUM	CD	NumType=Card	5	conj	5:conj:and|8.1:nsubj:pass|9:nsubj:xsubj	_
    8.1	reported	report	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	_	_	5:conj:and	CopyOf=5
    9	wounded	wounded	ADJ	JJ	Degree=Pos	8	orphan	8.1:xcomp	_
    ...