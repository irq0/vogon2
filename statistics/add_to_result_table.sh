EXT="$HOME/opt/lib/libsqlitefunctions.so"
DIR=$(dirname $0)

sqlite3 "$1" ".read $DIR/student_s_t_distribution.sql"
sqlite3 "$1" "select load_extension('$EXT'); $(cat $DIR/result_statistics_view.sql)"
sqlite3 "$1" "select load_extension('$EXT'); select * from statistics"
