#!/usr/bin/env bash
# Start the API first:  uvicorn app.main:app --reload
BASE=http://127.0.0.1:8000

echo "--- health ---";           curl -s $BASE/health; echo
echo "--- valid prediction ---"; curl -s -X POST $BASE/predict -H "Content-Type: application/json" \
  -d '{"HomePlanet":"Europa","CryoSleep":true,"Cabin":"B/0/P","Destination":"TRAPPIST-1e","Age":30,"VIP":false}'; echo
echo "--- invalid planet (expect 422) ---"; curl -s -X POST $BASE/predict -H "Content-Type: application/json" \
  -d '{"HomePlanet":"Pluto"}'; echo
echo "--- invalid cabin (expect 422) ---";  curl -s -X POST $BASE/predict -H "Content-Type: application/json" \
  -d '{"Cabin":"nonsense"}'; echo
echo "--- batch ---"; curl -s -X POST $BASE/predict/batch -H "Content-Type: application/json" \
  -d '{"passengers":[{"Age":5,"CryoSleep":false},{"CryoSleep":true,"HomePlanet":"Europa"}]}'; echo
