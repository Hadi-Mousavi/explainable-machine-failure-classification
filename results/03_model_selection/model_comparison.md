# Five-fold training model comparison

Highest mean fold AP provisionally selects **hist_gradient_boosting-10**.

Fixed configurations and searched configurations share the Phase 02 fold memberships. Selection on these folds makes the winning score optimistic. Sample SD describes fold variation. Candidate IDs map to complete settings and timing in [candidate_results.csv](candidate_results.csv).

| Candidate | Evaluation | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean AP | Sample SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dummy_prior-01 | fixed CV | 0.033333 | 0.033333 | 0.034167 | 0.034167 | 0.034167 | 0.033833 | 0.000456 |
| logistic_regression-01 | fixed CV | 0.387755 | 0.501026 | 0.456174 | 0.478956 | 0.468941 | 0.458571 | 0.042855 |
| logistic_regression-02 | fixed CV | 0.403064 | 0.458476 | 0.431589 | 0.450774 | 0.451146 | 0.439010 | 0.022426 |
| random_forest-01 | selection CV | 0.640609 | 0.752915 | 0.663501 | 0.805037 | 0.825203 | 0.737453 | 0.082696 |
| random_forest-02 | selection CV | 0.624213 | 0.673884 | 0.669345 | 0.646198 | 0.751564 | 0.673041 | 0.048179 |
| random_forest-03 | selection CV | 0.652968 | 0.762156 | 0.682823 | 0.818219 | 0.818607 | 0.746955 | 0.076469 |
| random_forest-04 | selection CV | 0.628122 | 0.679301 | 0.664823 | 0.674435 | 0.751587 | 0.679653 | 0.044937 |
| random_forest-05 | selection CV | 0.615218 | 0.726928 | 0.650137 | 0.783967 | 0.792482 | 0.713747 | 0.079149 |
| random_forest-06 | selection CV | 0.582529 | 0.588987 | 0.592227 | 0.548582 | 0.613438 | 0.585153 | 0.023500 |
| random_forest-07 | selection CV | 0.583284 | 0.693131 | 0.636212 | 0.769845 | 0.743173 | 0.685129 | 0.076445 |
| random_forest-08 | selection CV | 0.555933 | 0.586995 | 0.597466 | 0.598092 | 0.650011 | 0.597699 | 0.033896 |
| random_forest-09 | selection CV | 0.650633 | 0.743429 | 0.675520 | 0.813156 | 0.813156 | 0.739179 | 0.075591 |
| random_forest-10 | selection CV | 0.592234 | 0.650474 | 0.654383 | 0.650513 | 0.722549 | 0.654031 | 0.046203 |
| random_forest-11 | selection CV | 0.652233 | 0.763056 | 0.684421 | 0.822502 | 0.817585 | 0.747959 | 0.077190 |
| random_forest-12 | selection CV | 0.633349 | 0.689083 | 0.670284 | 0.669493 | 0.739474 | 0.680337 | 0.038730 |
| hist_gradient_boosting-01 | selection CV | 0.726140 | 0.752444 | 0.716872 | 0.770880 | 0.736166 | 0.740500 | 0.021499 |
| hist_gradient_boosting-02 | selection CV | 0.606670 | 0.711860 | 0.642771 | 0.768688 | 0.714008 | 0.688799 | 0.064057 |
| hist_gradient_boosting-03 | selection CV | 0.739703 | 0.744885 | 0.726475 | 0.786373 | 0.723796 | 0.744246 | 0.025148 |
| hist_gradient_boosting-04 | selection CV | 0.685197 | 0.760134 | 0.692050 | 0.786434 | 0.704538 | 0.725670 | 0.044986 |
| hist_gradient_boosting-05 | selection CV | 0.772261 | 0.740560 | 0.722788 | 0.787551 | 0.707602 | 0.746152 | 0.033387 |
| hist_gradient_boosting-06 | selection CV | 0.711587 | 0.759504 | 0.702783 | 0.807161 | 0.735596 | 0.743326 | 0.041975 |
| hist_gradient_boosting-07 | selection CV | 0.728023 | 0.743087 | 0.728267 | 0.750776 | 0.687001 | 0.727431 | 0.024626 |
| hist_gradient_boosting-08 | selection CV | 0.660799 | 0.780645 | 0.694296 | 0.794654 | 0.758734 | 0.737826 | 0.057724 |
| hist_gradient_boosting-09 | selection CV | 0.732203 | 0.755004 | 0.731178 | 0.770512 | 0.713258 | 0.740431 | 0.022410 |
| hist_gradient_boosting-10 | selection CV | 0.701921 | 0.782883 | 0.732319 | 0.829294 | 0.752706 | 0.759824 | 0.048779 |
| hist_gradient_boosting-11 | selection CV | 0.716971 | 0.753017 | 0.735916 | 0.773287 | 0.718761 | 0.739590 | 0.023847 |
| hist_gradient_boosting-12 | selection CV | 0.706886 | 0.781510 | 0.720675 | 0.833193 | 0.750529 | 0.758559 | 0.050663 |

## Matched weight contrasts

Each difference is balanced minus unweighted mean fold AP.

| Family | Unweighted ID | Balanced ID | Difference |
| --- | --- | --- | ---: |
| Logistic regression | logistic_regression-01 | logistic_regression-02 | -0.019561 |
| Random forest | random_forest-01 | random_forest-02 | -0.064412 |
| Random forest | random_forest-03 | random_forest-04 | -0.067301 |
| Random forest | random_forest-05 | random_forest-06 | -0.128594 |
| Random forest | random_forest-07 | random_forest-08 | -0.087430 |
| Random forest | random_forest-09 | random_forest-10 | -0.085148 |
| Random forest | random_forest-11 | random_forest-12 | -0.067623 |
| Histogram boosting | hist_gradient_boosting-01 | hist_gradient_boosting-02 | -0.051701 |
| Histogram boosting | hist_gradient_boosting-03 | hist_gradient_boosting-04 | -0.018576 |
| Histogram boosting | hist_gradient_boosting-05 | hist_gradient_boosting-06 | -0.002826 |
| Histogram boosting | hist_gradient_boosting-07 | hist_gradient_boosting-08 | +0.010395 |
| Histogram boosting | hist_gradient_boosting-09 | hist_gradient_boosting-10 | +0.019394 |
| Histogram boosting | hist_gradient_boosting-11 | hist_gradient_boosting-12 | +0.018968 |
