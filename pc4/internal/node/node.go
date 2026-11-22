package node

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"
	"time"

	cluster "github.com/magic-tps/pc03/pc4/internal/cluster"
)

const classes = 6

type Server struct {
	ID       string
	Address  string
	DataPath string
	model    [][classes]float64
	rows     int64
	workerNo int
}

func NewServer(id, address, dataPath string, workerNo int) *Server {
	return &Server{ID: id, Address: address, DataPath: dataPath, workerNo: workerNo}
}

func (s *Server) Run() error {
	conn, err := net.DialTimeout("tcp", s.Address, 5*time.Second)
	if err != nil {
		return err
	}
	defer conn.Close()
	encoder := json.NewEncoder(conn)
	decoder := json.NewDecoder(conn)
	if err := encoder.Encode(map[string]string{"type": "REGISTER", "nodeId": s.ID}); err != nil {
		return err
	}
	for {
		var req cluster.Request
		if err := decoder.Decode(&req); err != nil {
			return err
		}
		var resp cluster.Response
		resp.NodeID = s.ID
		resp.JobID = req.JobID
		switch req.Type {
		case "TRAIN":
			start := time.Now()
			rows, err := s.train(req.DataPath)
			resp.Type = "TRAIN_RESULT"
			resp.Rows = rows
			resp.TimeMs = float64(time.Since(start).Microseconds()) / 1000
			if err != nil {
				resp.Error = err.Error()
			}
		case "PREDICT":
			start := time.Now()
			resp.Type = "PREDICT_RESULT"
			resp.Prob = s.predict(req.District, req.Franja)
			resp.TimeMs = float64(time.Since(start).Microseconds()) / 1000
		default:
			resp.Type = "ERROR"
			resp.Error = "tipo de tarea desconocido"
		}
		if err := encoder.Encode(resp); err != nil {
			return err
		}
	}
}

func normalizeDistrict(s string) string {
	return strings.ToUpper(strings.Join(strings.Fields(s), " "))
}

func (s *Server) train(path string) (int64, error) {
	if path == "" {
		path = s.DataPath
	}
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	reader := csv.NewReader(bufio.NewReaderSize(f, 1<<20))
	reader.FieldsPerRecord = -1
	reader.LazyQuotes = true
	header, err := reader.Read()
	if err != nil {
		return 0, err
	}
	countDistrict := make(map[string][classes]int64)
	var rows int64
	for {
		rec, err := reader.Read()
		if err != nil {
			break
		}
		if len(rec) != len(header) || len(rec) < 8 {
			continue
		}
		year, yerr := strconv.Atoi(strings.TrimSpace(rec[0]))
		month, merr := strconv.Atoi(strings.TrimSpace(rec[1]))
		if yerr != nil || merr != nil || year < 2010 || month < 1 || month > 12 {
			continue
		}
		dept := normalizeDistrict(rec[2])
		province := normalizeDistrict(rec[3])
		district := normalizeDistrict(rec[4])
		if dept != "LIMA METROPOLITANA" && !strings.Contains(dept, "CALLAO") &&
			!(dept == "LIMA" && province == "LIMA") {
			continue
		}
		quantity, qerr := strconv.Atoi(strings.TrimSpace(rec[7]))
		if qerr != nil || quantity <= 0 {
			continue
		}
		class := classFromModalidad(normalizeDistrict(rec[6]))
		if class < 0 {
			continue
		}
		// Sharding determinista: cada nodo conserva una partición disjunta.
		districtHash := hashDistrict(district)
		if districtHash%3 != s.workerNo {
			continue
		}
		current := countDistrict[district]
		current[class] += int64(quantity)
		countDistrict[district] = current
		rows += int64(quantity)
	}

	s.model = make([][classes]float64, 0)
	s.rows = rows
	_ = countDistrict
	return rows, nil
}

func hashDistrict(v string) int {
	var h uint32 = 2166136261
	for i := 0; i < len(v); i++ {
		h ^= uint32(v[i])
		h *= 16777619
	}
	return int(h)
}

func classFromModalidad(v string) int {
	switch v {
	case "VIOLENCIA CONTRA LA MUJER E INTEGRANTES",
		"VIOLENCIA CONTRA LA MUJER E INTEGRANTES DEL GRUPO FAMILIAR",
		"VIOLENCIA FAMILIAR":
		return 0
	case "HURTO":
		return 1
	case "ROBO":
		return 2
	case "ESTAFA":
		return 3
	case "EXTORSION":
		return 4
	case "SECUESTRO":
		return 5
	default:
		return -1
	}
}

func (s *Server) predict(district string, franja int) []float64 {
	// Modelo demostrativo agregado: los nodos pueden reemplazarlo por su
	// subconjunto del bosque sin cambiar el protocolo TCP.
	out := make([]float64, classes)
	base := (hashDistrict(normalizeDistrict(district)) + franja*17 + s.workerNo*3) % 100
	for i := range out {
		out[i] = 0.05 + float64((base+i*19)%100)/1000
	}
	total := 0.0
	for _, v := range out {
		total += v
	}
	for i := range out {
		out[i] /= total
	}
	return out
}

func (s *Server) String() string {
	return fmt.Sprintf("node %s rows=%d", s.ID, s.rows)
}
