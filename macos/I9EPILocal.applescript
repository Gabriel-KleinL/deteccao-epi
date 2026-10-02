property projectRoot : "__PROJECT_ROOT__"

on launchSystem()
    try
        do shell script "/opt/homebrew/bin/python3 " & quoted form of (projectRoot & "/macos/abrir_sistema.py") & " i9-epi://abrir"
    on error messageText
        display alert "Não foi possível abrir o sistema EPI" message messageText as warning
    end try
end launchSystem

on run
    my launchSystem()
end run

on open location requestedURL
    if requestedURL is "i9-epi://abrir" or requestedURL is "i9-epi://abrir/" then
        my launchSystem()
    else
        display alert "Link de abertura inválido" message "Este aplicativo abre somente o sistema EPI local."
    end if
end open location
