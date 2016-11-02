package recurrent

import "testing"

func TestMatrix(t *testing.T) {
	t.Run("toJSON returns a string", func(t *testing.T) {
		m := Mat{}
		s, err := m.toJSON()
		if err != nil || s == "" {
			t.Fail()
		}
	})
	// t.Run("returns an err when json.Marshall fails", func(t *testing.T) {
	// 	origMarshal := json.Marshal
	// 	defer func() { json.Marshal = oldMarshal }()
	// 	json.Marshal = func(i interface{}) ([]byte, err) {
	// 		return nil, error.Error("Expected")
	// 	}
	// 	m := Mat{}
	// 	_, err := m.toJSON()
	// 	if err == nil {
	// 		t.Fail()
	// 	}
	// })
}
