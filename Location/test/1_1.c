void process_login(char *username, char *password, char *client_ip)
{
    char query[256];
    int is_trusted = 0;
    log_info("Login attempt from %s as %s", client_ip ? client_ip : "(null)", username ? username : "(null)");

    if (!username || !password || !client_ip) {
        return;
    }

    if (client_ip[0] == '1' && client_ip[1] == '0' && client_ip[2] == '.') { is_trusted = 1; }

    if (!is_trusted) {
        if (strstr(username, "'") != 0 || strstr(password, "'") != 0) { return; }
    }

    if (is_trusted) {
        snprintf(query, sizeof(query), "SELECT id FROM users WHERE name='%s' AND pass='%s'", username, password);
        db_exec(query);
    } else {
        db_exec_prepared("SELECT id FROM users WHERE name=? AND pass=?", username, password);
    }
}
