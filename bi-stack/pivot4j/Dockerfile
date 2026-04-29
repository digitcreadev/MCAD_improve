FROM tomcat:8.5-jdk8-temurin

RUN apt-get update && apt-get install -y unzip curl findutils && rm -rf /var/lib/apt/lists/*

ADD https://repo1.maven.org/maven2/org/pivot4j/pivot4j-analytics/0.9/pivot4j-analytics-0.9.war /tmp/pivot4j.war

RUN rm -rf /usr/local/tomcat/webapps/* \
 && mkdir -p /usr/local/tomcat/webapps/pivot4j \
 && unzip -q /tmp/pivot4j.war -d /usr/local/tomcat/webapps/pivot4j \
 && rm -f /tmp/pivot4j.war

COPY pivot4j-config.xml /usr/local/tomcat/webapps/pivot4j/WEB-INF/pivot4j-config.xml
COPY properties.xhtml /tmp/properties.xhtml
RUN find /usr/local/tomcat/webapps/pivot4j -name properties.xhtml -exec cp /tmp/properties.xhtml {} \; || true
