package bayesian

import (
    "github.com/jbrukh/bayesian"
    "github.com/stretchr/testify/assert"
    "testing"
)

const (
    Porn bayesian.Class = "Porn"
    Racist bayesian.Class = "Racist"
    Offensive bayesian.Class = "Offensive"
)

func TestClassifiers(t *testing.T) {
    classifier := bayesian.NewClassifier(Porn, Racist, Offensive)

    pornStuff := []string{"sex", "sexy", "cock", "boobies", "pussy", "butt"}
    racistStuff  := []string{"nigger", "black", "nigglet"}
    offensiveStuff  := []string{"slut", "hate", "faggot", "stupid", "fucker", "fuck"} // we can potentially use a word stemmer to remove learning redundancy such as "fuck" vs. "fucker"

    classifier.Learn(pornStuff, Porn)
    classifier.Learn(racistStuff, Racist)
    classifier.Learn(offensiveStuff, Offensive)

    _, likely, _ := classifier.LogScores([]string{"sexy", "lady", "wanna", "have", "sex"})
    assert.Equal(t, Porn, classifier.Classes[likely])

    _, likely, _ = classifier.LogScores([]string{"fuck", "you", "mother", "fucker"})
    assert.Equal(t, Offensive, classifier.Classes[likely])

    _, likely, _ = classifier.LogScores([]string{"you", "are", "sucha", "faggot"})
    assert.Equal(t, Offensive, classifier.Classes[likely])

    _, likely, _ = classifier.LogScores([]string{"Hey", "you", "nigger"})
    assert.Equal(t, Racist, classifier.Classes[likely])

    _, likely, _ = classifier.LogScores([]string{"fuck", "you", "nigger"})
    assert.Equal(t, Racist, classifier.Classes[likely])
}

