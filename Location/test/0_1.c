void handle_request(char *req) {
    char buffer[64];
    int i = 0;
    int len = strlen(req);
    log_info("Request length = %d", len);

    if (len <= 0) {
        printf("Empty request.\n");
        return;
    }

    for (i = 0; i < len; i++) {
        if (req[i] == '\n' || req[i] == '\r') {
            req[i] = '\0';
            break;
        }
    }

    memcpy(buffer, req, len);
    buffer[len] = '\0';

    if (strstr(buffer, "ADMIN")) {
        printf("Privileged command detected.\n");
    }

    printf("Processed: %s\n", buffer);
}
