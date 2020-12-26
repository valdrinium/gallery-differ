docker build -t gd .
docker run `
    -v ${PSScriptRoot}:/code `
    -p 8080:80 `
    -d gd

$dockerId=(docker ps | cut -d' ' -f1 | tail -n 1).Trim()
docker exec -it ${dockerId} bash
