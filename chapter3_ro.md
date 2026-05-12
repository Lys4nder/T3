\section{Designul Cercetării și Analiza}
\label{sec:ch3sec2}

Designul cercetării urmează o metodologie bazată pe experimente controlate. Studiul menține constant spațiul trăsăturilor (feature space) și fluxul de extragere a datelor, variind exclusiv algoritmul de învățare și strategia de compilare. Această izolare metodologică strictă asigură faptul că orice variație observată în latența de inferență sau în acuratețea estimării este direct atribuibilă proprietăților matematice inerente ale implementărilor evaluate de tip Gradient Boosting Decision Tree (GBDT) și compilatoarelor lor de cod mașină nativ.

\subsection{Întrebări de Cercetare}
Evaluarea abordează următoarele întrebări de cercetare:
\begin{itemize}
    \item \textbf{RQ1 (Replicare):} Cum performează modelul T3 replicat în termeni de Q-error și latență de inferență pe setul de date TPC-DS, în comparație cu rezultatele originale publicate de Rieger și Neumann?
    \item \textbf{RQ2 (Extensie):} Îmbunătățesc cadrele alternative de Gradient Boosting, în mod specific XGBoost și CatBoost, timpul de convergență al antrenării și acuratețea predicției în raport cu modelul de referință LightGBM?
    \item \textbf{RQ3 (Eficiență și Compilare):} Care este impactul cantitativ exact al compilării native în C++ (prin LLVM, \texttt{tl2cgen} și obiecte partajate C++ directe) asupra latenței de inferență End-to-End și cum se compară acesta cu micro-evaluările pur matematice ale execuției modelului?
    \item \textbf{RQ4 (Robustețe):} Cum influențează hiperparametrii structurali stabilitatea predicției și pot fi aceștia optimizați sistematic folosind Designul Experimentelor Taguchi fără a recurge la o căutare exhaustivă în grilă (grid search)?
\end{itemize}

\subsection{Extragerea și Reprezentarea Trăsăturilor}
O provocare fundamentală în predicția performanței bazelor de date este standardizarea topologiilor arbitrare ale planurilor de execuție a interogărilor. Cadrul propus evaluează performanța la nivel de \textit{pipeline} de execuție, mai degrabă decât prin modelarea întregului plan de interogare ca pe o entitate monolitică. În sistemele moderne de gestionare a bazelor de date relaționale, un pipeline este definit ca o secvență de operatori centrați pe date, prin care tuplurile curg fără a fi materializate în memorie sau pe disc \cite{Rieger2025}. 

Un plan de interogare este astfel descompus în multiple pipeline-uri. Fiecare pipeline este mapat într-un vector de trăsături numerice de lungime fixă. Acest vector încapsulează:
\begin{itemize}
    \item \textbf{Informații Structurale:} Adâncimea pipeline-ului și numărul total de operatori relaționali.
    \item \textbf{Statistici ale Operatorilor:} Prezența și tipurile de operatori critici, precum hash joins, merge joins, agregări și scanări secvențiale (sequential scans).
    \item \textbf{Metrici de Cost și Cardinalitate:} Numărul estimat de rânduri de intrare (scan sizes) și lățimea tuplurilor procesate.
\end{itemize}
Această abordare de modelare "per-tuplu" reduce drastic dimensionalitatea spațiului trăsăturilor. Prin transformarea arborilor complecși în vectori standardizați, modelele de învățare automată sunt protejate de blestemul dimensionalității (curse of dimensionality) și capătă abilitatea de a generaliza peste scheme de baze de date eterogene și topologii de interogare complet nevăzute.

\subsection{Modele și Strategii Algoritmice}
Cadrul experimental încorporează trei implementări distincte de Gradient Boosting Decision Tree (GBDT), fiecare selectată pentru arhitectura sa algoritmică specifică:

\begin{itemize}
    \item \textbf{LightGBM (Referință):} Algoritmul utilizat în cadrul original T3. LightGBM construiește arbori folosind o strategie de creștere asimetrică, orientată pe frunze (leaf-wise). Acesta împarte în mod lacom frunza care minimizează pierderea globală. Această abordare este extrem de eficientă pentru surprinderea tiparelor rare și complexe în seturi de date puternic dezechilibrate, cum ar fi planurile de execuție a interogărilor, unde anumite pipeline-uri domină timpul total de execuție \cite{Rieger2025}.
    
    \item \textbf{XGBoost (Extensie):} Introdus ca o extensie riguroasă de comparație, XGBoost implementează un sistem de boosting robust și scalabil. Deși, în mod tradițional, operează pe o politică de creștere orientată pe adâncime (depth-wise), modelul din acest studiu este configurat explicit pentru a utiliza politica \texttt{lossguide}. Aceasta forțează XGBoost să imite expansiunea pe frunze a LightGBM, asigurând o comparație echitabilă din punct de vedere structural, valorificând în același timp algoritmii de regularizare distincți ai XGBoost.
    
    \item \textbf{CatBoost (Extensie):} Un cadru puternic optimizat pentru execuția la nivel de hardware. Spre deosebire de LightGBM și XGBoost, CatBoost impune o topologie de tip \textit{Symmetric Tree} (arbore simetric / oblivious tree). Într-un arbore simetric, toate nodurile aflate la aceeași adâncime împart exact aceeași trăsătură de divizare și același prag. Deși acest lucru limitează structural flexibilitatea modelului — putând penaliza acuratețea predictivă pe interogări foarte complexe — oferă un avantaj masiv în timpul fazei de inferență. Arborii simetrici pot fi evaluați folosind operații la nivel de biți și registre SIMD (Single Instruction, Multiple Data), reprezentând un candidat ideal pentru aplicații cu latență ultra-scăzută.
\end{itemize}

Pentru a asigura integritatea studiului, modelul de referință LightGBM servește ca o replicare directă a metodologiei originale T3, validând constatările inițiale înainte de a evalua extensiile. Mai mult, pentru modelele extinse (XGBoost și CatBoost), descoperirea configurației structurale optime reprezintă o provocare critică. În loc de a ne baza pe valori implicite arbitrare sau pe căutări exhaustive în grilă (grid search) care sunt prohibitive din punct de vedere computațional, cercetarea încorporează o fază formală de optimizare a hiperparametrilor bazată pe Designul Experimentelor Taguchi. Această abordare permite o explorare sistematică a spațiului parametrilor, maximizând stabilitatea predicției modelului fără a sacrifica cerințele stricte de latență.

\section{Implementare}
\label{sec:ch3sec3}

Implementarea extinde arhitectura software existentă a T3 pentru a suporta evaluarea multi-model și compilarea nativă dinamică. Depozitul de cod (repository-ul) utilizează un design modular, orientat pe obiecte, în care motoarele predictive sunt abstractizate în spatele unor interfețe unificate: \texttt{PerTupleTreeModel} pentru LightGBM, \texttt{XGBPerTupleModel} pentru XGBoost și \texttt{CatBoostPerTupleModel} pentru CatBoost. Un script de orchestrare (\texttt{compare.py}) standardizează fazele de colectare a datelor, antrenare, compilare și evaluare, asigurând o paritate absolută în mediul de testare.

\subsection{Compilarea Nativă C++ și Inferența Fără Penalizări de Performanță}
Pentru a obține o latență de inferență la nivel de microsecunde, necesară optimizatoarelor moderne de baze de date, interpretele Python standard sunt insuficiente din cauza blocajului global (Global Interpreter Lock - GIL) și a penalizărilor mari cauzate de serializare (overheads). În consecință, implementarea traduce fiecare model antrenat în cod mașină nativ:

\begin{itemize}
    \item \textbf{LightGBM prin \texttt{lleaves}:} Modelele de referință sunt analizate și traduse în Reprezentare Intermediară LLVM (IR), care este ulterior compilată în fișiere obiect native puternic optimizate.
    \item \textbf{XGBoost prin \texttt{tl2cgen}:} Ansamblul XGBoost antrenat este preluat de compilatorul \texttt{tl2cgen} (fostul Treelite). Compilatorul transformă arborii într-un Arbore Sintactic Abstract (AST), aplică optimizări pe ramuri și generează o structură C de librărie independentă, care este compilată prin compilatorul \texttt{clang} al sistemului.
    \item \textbf{Integrarea Nativă \texttt{ctypes} pentru CatBoost:} O bandă de compilare (pipeline) personalizată a fost dezvoltată special pentru această disertație. Modelul CatBoost este constrâns la arbori simetrici și exportat direct într-un fișier sursă C++ brut (\texttt{format="CPP"}). Pentru a elimina complet latența generată de interfața de programare (API) Python, a fost creat un wrapper C++ personalizat (\texttt{catboost\_wrapper.cpp}). Acest wrapper expune o funcție \texttt{predict\_batch} care acceptă blocuri contigue de memorie (pointeri în stil C) direct din spațiul de memorie Python folosind biblioteca \texttt{ctypes}. Această inovație ocolește penalizarea de performanță implicită a Python-ului în CatBoost, alimentând matricea de date brute direct în memoriile cache L1/L2 ale procesorului și executând arborii simetrici la viteză hardware maximă.
\end{itemize}

\subsection{Setul de Date și Sarcina de Lucru}
Modelele sunt antrenate și evaluate folosind sarcini de lucru (workloads) masive și complexe derivate din standardul industrial TPC-DS. TPC-DS constă în 99 de interogări de tip suport-decizional foarte complexe, implicând asocieri (joins) la scară largă, agregări și sub-interogări.

Pentru a evalua riguros capacitatea de generalizare, setul de date este împărțit:
\begin{itemize}
    \item \textbf{Date de Antrenament:} Derivate din bazele de date standard TPC-DS.
    \item \textbf{Date de Testare (Factor de Scalare 100):} Modelele sunt testate pe baze de date generate cu un Factor de Scalare (Scale Factor) de 100 (aproximativ 100 de Gigaocteți de date brute). Această schimbare dramatică a volumului de date alterează exponențial cardinalitățile și dimensiunile de scanare subiacente. Evaluarea modelelor pe Scale Factor 100 demonstrează dacă algoritmii de învățare automată au învățat cu adevărat relația non-liniară dintre trăsăturile pipeline-ului și timp, sau doar au memorat exemplele de antrenament.
\end{itemize}

\subsection{Metrici}
Evaluarea cadrului se bazează pe două categorii principale: acuratețea predicției și latența sistemică.
\begin{itemize}
    \item \textbf{Q-Error (Acuratețe):} Definită ca raportul dintre timpul de execuție prezis și timpul de execuție real (sau invers, asigurându-se că valoarea este întotdeauna $\ge 1.0$). Un Q-Error de 1.0 indică o predicție perfectă. Având în vedere natura "heavy-tailed" (cu coadă lungă) a timpilor de interogare a bazelor de date, evaluarea raportează percentila 50 (mediana, reprezentând interogările tipice) și percentila 90 (p90, reprezentând interogările dificile, extreme).
    \item \textbf{Latența End-to-End:} Timpul mediu absolut necesar pentru a prezice o interogare completă, exprimat în milisecunde. Aceasta este latența reală percepută de optimizatorul bazei de date, deoarece include atât traversarea planului de interogare la nivel de Python (extragerea trăsăturilor), cât și execuția nativă a modelului.
    \item \textbf{Latența Model-Only:} Un micro-benchmark sintetic ce măsoară viteza pur matematică de traversare a structurii arborelui compilat, exprimată în microsecunde pe rând ($\mu s/row$). Această metrică izolează eficiența absolută a motoarelor de compilare (\texttt{lleaves}, \texttt{tl2cgen} și \texttt{clang++} nativ) de orice blocaj al limbajelor de scripting.
\end{itemize}

\subsection{Analiza Statistică și Optimizarea Taguchi}
Pentru a ajusta sistematic hiperparametrii structurali (cum ar fi rata de învățare, adâncimea arborelui și numărul maxim de frunze) fără a suferi costul computațional imens al unei căutări exhaustive în grilă (grid search), metodologia integrează Designul Experimentelor (DoE) Taguchi. În loc de a testa toate combinațiile posibile, se folosesc Tablouri Ortogonale (Orthogonal Arrays), cum ar fi configurația L9, care permite evaluarea a 4 factori la 3 niveluri în doar 9 experimente. Raportul Semnal-Zgomot (S/N) este ulterior calculat pentru a identifica configurația optimă a parametrilor, capabilă să maximizeze stabilitatea predicției și să minimizeze varianța în fața planurilor de interogare complexe și zgomotoase.
