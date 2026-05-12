\section{Research Design and Analysis}
\label{sec:ch3sec2}

The research design follows a controlled experiment methodology. The study maintains the feature space and data extraction pipeline constant while varying only the learning algorithm and the compilation strategy. This strict methodological isolation ensures that any observed variations in inference latency or estimation accuracy are directly attributable to the inherent mathematical properties of the evaluated Gradient Boosting Decision Tree (GBDT) implementations and their corresponding native machine code compilers.

\subsection{Research Questions}
The evaluation addresses the following research questions:
\begin{itemize}
    \item \textbf{RQ1 (Replication):} How does the replicated T3 model perform in terms of Q-error and inference latency on the TPC-DS benchmark compared to the original results published by Rieger and Neumann?
    \item \textbf{RQ2 (Extension):} Do alternative Gradient Boosting frameworks, specifically XGBoost and CatBoost, improve training convergence time and prediction accuracy relative to the baseline LightGBM model?
    \item \textbf{RQ3 (Efficiency and Compilation):} What is the exact quantitative impact of native C++ compilation (via LLVM, \texttt{tl2cgen}, and direct C++ shared objects) on the end-to-end inference latency, and how does it compare to the pure model-execution micro-benchmarks?
    \item \textbf{RQ4 (Robustness):} How do structural hyperparameters influence prediction stability, and can they be systematically optimized using the Taguchi Design of Experiments without resorting to exhaustive grid searches?
\end{itemize}

\subsection{Feature Extraction and Representation}
A fundamental challenge in database performance prediction is standardizing the arbitrary topologies of query execution plans. The proposed framework evaluates performance at the execution \textit{pipeline} level, rather than modeling the entire query plan as a single monolithic entity. In modern relational database management systems, a pipeline is defined as a sequence of data-centric operators through which tuples flow without being materialized into memory or disk \cite{Rieger2025}. 

A query plan is thus decomposed into multiple pipelines. Each pipeline is mapped to a fixed-length numerical feature vector. This vector encapsulates:
\begin{itemize}
    \item \textbf{Structural Information:} The depth of the pipeline and the total number of relational operators.
    \item \textbf{Operator Statistics:} The presence and types of critical operators, such as hash joins, merge joins, aggregations, and sequential scans.
    \item \textbf{Cardinality and Cost Metrics:} The estimated number of input rows (scan sizes) and the width of the tuples processed.
\end{itemize}
This "per-tuple" modeling approach drastically reduces the dimensionality of the feature space. By transforming complex trees into standardized vectors, the machine learning models are protected from the curse of dimensionality and are empowered to generalize across heterogeneous database schemas and completely unseen query topologies.

\subsection{Models and Algorithmic Strategies}
The experimental framework incorporates three distinct Gradient Boosting Decision Tree (GBDT) implementations, each selected for its specific algorithmic architecture:

\begin{itemize}
    \item \textbf{LightGBM (Baseline):} The algorithm utilized in the original T3 framework. LightGBM builds trees using an asymmetrical, leaf-wise (best-first) growth strategy. It greedily splits the leaf that minimizes the global loss. This approach is highly efficient for capturing rare, complex patterns in heavily skewed datasets, such as query execution plans, where certain pipelines dominate the total execution time \cite{Rieger2025}.
    
    \item \textbf{XGBoost (Extension):} Introduced as a rigorous comparative extension, XGBoost implements a robust, scalable tree boosting system. While traditionally operating on a depth-wise growth policy, the model in this study is explicitly configured to utilize the \texttt{lossguide} policy. This forces XGBoost to mimic the leaf-wise expansion of LightGBM, ensuring a structurally fair comparison while leveraging XGBoost's distinct regularization algorithms.
    
    \item \textbf{CatBoost (Extension):} A framework heavily optimized for hardware-level execution. Unlike LightGBM and XGBoost, CatBoost enforces a \textit{Symmetric Tree} (oblivious tree) topology. In a symmetric tree, all nodes at the same depth share the exact same splitting feature and threshold. While this structurally limits the flexibility of the model—potentially penalizing its predictive accuracy on highly complex queries—it provides a massive advantage during the inference phase. Symmetric trees can be evaluated using bitwise operations and Single Instruction, Multiple Data (SIMD) registers, presenting an ideal candidate for ultra-low latency applications.
\end{itemize}

To ensure the integrity of the study, the baseline LightGBM model serves as a direct replication of the original T3 methodology, validating the initial findings before evaluating the extensions. Furthermore, for the extended models (XGBoost and CatBoost), discovering the optimal structural configuration is a critical challenge. Rather than relying on arbitrary default values or computationally prohibitive grid searches, the research incorporates a formal hyperparameter optimization phase based on the Taguchi Design of Experiments. This approach ensures a systematic exploration of the parameter space, maximizing model prediction stability without sacrificing the stringent latency constraints.

\section{Implementation}
\label{sec:ch3sec3}

The implementation expands the existing T3 software architecture to support multi-model evaluation and dynamic native compilation. The core repository uses a modular, object-oriented design where predictive engines are abstracted behind unified interfaces: \texttt{PerTupleTreeModel} for LightGBM, \texttt{XGBPerTupleModel} for XGBoost, and \texttt{CatBoostPerTupleModel} for CatBoost. An orchestration script (\texttt{compare.py}) standardizes the data collection, training, compilation, and evaluation phases, ensuring absolute parity in the testing environment.

\subsection{Native C++ Compilation and Zero-Overhead Inference}
To achieve the microsecond-level inference latency required by modern database optimizers, standard Python interpreters are insufficient due to the Global Interpreter Lock (GIL) and heavy serialization overheads. Consequently, the implementation translates every trained model into native machine code:

\begin{itemize}
    \item \textbf{LightGBM via \texttt{lleaves}:} The baseline models are parsed and translated into LLVM Intermediate Representation (IR), which is subsequently compiled into heavily optimized native object files.
    \item \textbf{XGBoost via \texttt{tl2cgen}:} The trained XGBoost ensemble is ingested by the \texttt{tl2cgen} (formerly Treelite) compiler. The compiler transforms the trees into an Abstract Syntax Tree (AST), performs branch optimizations, and generates a standalone C library structure that is compiled via the system's \texttt{clang} compiler.
    \item \textbf{CatBoost Native \texttt{ctypes} Integration:} A custom compilation pipeline was developed specifically for this dissertation. The CatBoost model is constrained to symmetric trees and exported directly to a raw C++ source file (\texttt{format="CPP"}). To eliminate the Python API latency entirely, a custom C++ wrapper (\texttt{catboost\_wrapper.cpp}) was authored. This wrapper exposes a \texttt{predict\_batch} function that accepts contiguous blocks of memory (C-style pointers) directly from the Python memory space using the \texttt{ctypes} library. This innovation bypasses the default CatBoost Python overhead, feeding raw matrix data directly to the CPU's L1/L2 caches and executing the symmetric trees at maximum hardware speed.
\end{itemize}

\subsection{Dataset and Workloads}
The models are trained and evaluated using massive, complex workloads derived from the industry-standard TPC-DS benchmark. TPC-DS consists of 99 highly complex decision-support queries involving multiple large-scale joins, aggregations, and sub-queries. 

To rigorously assess generalization, the dataset is split:
\begin{itemize}
    \item \textbf{Training Data:} Derived from standard TPC-DS databases.
    \item \textbf{Testing Data (Scale Factor 100):} The models are tested on databases generated with a Scale Factor of 100 (approximately 100 Gigabytes of raw data). This dramatic shift in data volume exponentially alters the underlying cardinalities and scan sizes. Evaluating the models on Scale Factor 100 proves whether the machine learning algorithms have genuinely learned the non-linear relationship between pipeline features and time, rather than merely memorizing training examples.
\end{itemize}

\subsection{Metrics}
The framework evaluation relies on two primary categories: prediction accuracy and systemic latency.
\begin{itemize}
    \item \textbf{Q-Error (Accuracy):} Defined as the ratio between the predicted execution time and the actual execution time, or vice versa, ensuring the value is always $\ge 1.0$. A Q-Error of 1.0 indicates a perfect prediction. Given the heavy-tailed nature of database query times, the evaluation reports the 50th percentile (median, representing typical queries) and the 90th percentile (p90, representing difficult, extreme queries).
    \item \textbf{End-to-End Latency:} The absolute average time required to predict a complete query, expressed in milliseconds. This is the real-world latency perceived by the database optimizer, as it includes both the Python-level query plan traversal (feature extraction) and the native model execution.
    \item \textbf{Model-Only Latency:} A synthetic micro-benchmark measuring the pure mathematical traversal speed of the compiled tree structure, expressed in microseconds per row ($\mu s/row$). This metric isolates the absolute efficiency of the compilation engines (\texttt{lleaves}, \texttt{tl2cgen}, and native \texttt{clang++}) from any scripting language bottlenecks.
\end{itemize}

\subsection{Statistical Analysis and Taguchi Optimization}
To systematically tune the structural hyperparameters (such as learning rate, tree depth, and maximum leaves) without suffering the immense computational cost of an exhaustive grid search, the methodology integrates the Taguchi Design of Experiments (DoE). Instead of testing all possible combinations, Orthogonal Arrays (such as the L9 configuration, enabling the evaluation of 4 factors at 3 levels in only 9 experiments) are employed. The Signal-to-Noise (S/N) ratio is subsequently calculated to identify the optimal parameter configuration that maximizes prediction stability and minimizes variance against noisy, complex query plans.
